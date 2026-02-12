"""
pgai CSV Data Explorer — FastAPI Backend
=========================================
Bridges the CSV Data Explorer frontend to PostgreSQL + OpenAI SQL-RAG.

Endpoints:
    GET  /health       — Check PostgreSQL connection
    POST /upload       — Upload CSV, create table in csvdata schema
    POST /query        — SQL-RAG: LLM generates SQL → execute → LLM formats answer

Usage:
    cd projects/pgai
    uv run python ../../csv_server.py
"""

import asyncio
import csv
import io
import os
import re
import sys

from pathlib import Path

# Load .env from project root before any other imports use env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import openai
import psycopg
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Windows async compatibility (per projects/pgai examples)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DB_URL = os.getenv("DB_URL", "postgresql://postgres@localhost:5432/postgres")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
SCHEMA = "csvdata"

# OpenAI client (uses OPENAI_API_KEY and OPENAI_BASE_URL env vars automatically)
oai_client = openai.AsyncOpenAI()

app = FastAPI(title="pgai CSV Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track all loaded tables: {table_name: {columns, types, rows, filename}}
loaded_tables: dict[str, dict] = {}

# Path to the HTML frontend
HTML_PATH = Path(__file__).parent / "csv-query.html"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the CSV Data Explorer frontend."""
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


# ── Helpers ──────────────────────────────────────────────────────────────────


def sanitize_name(name: str) -> str:
    """Turn an arbitrary string into a safe PostgreSQL identifier."""
    s = re.sub(r"[^\w]", "_", name.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "col_" + s
    return s[:63]  # PG identifier max length


def infer_pg_type(values: list[str], col_name: str = "") -> str:
    """Guess a PostgreSQL column type from sample values."""
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "text"
    # ID columns are always text — they may look numeric in one table
    # but be alphanumeric in another, causing JOIN type mismatches.
    if col_name.endswith("_id") or col_name == "id":
        return "text"
    # Try numeric — require ALL sampled values to be numeric
    numeric_count = 0
    has_dot = False
    sample = non_empty[:100]
    for v in sample:
        cleaned = v.replace(",", "").strip()
        try:
            float(cleaned)
            numeric_count += 1
            if "." in cleaned:
                has_dot = True
        except ValueError:
            pass
    if numeric_count == len(sample):
        return "double precision" if has_dot else "bigint"
    return "text"


# ── Health ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Check PostgreSQL connection."""
    try:
        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            await (await con.execute("SELECT 1")).fetchone()
            return {
                "status": "connected",
                "database": DB_URL.split("/")[-1].split("?")[0],
                "tables": list(loaded_tables.keys()),
            }
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


@app.get("/tables")
async def list_tables():
    """List all loaded CSV tables with metadata."""
    return {
        name: {
            "filename": info["filename"],
            "columns": info["columns"],
            "rows": info["rows"],
        }
        for name, info in loaded_tables.items()
    }


@app.delete("/tables/{table_name}")
async def delete_table(table_name: str):
    """Drop a loaded CSV table."""
    if table_name not in loaded_tables:
        return {"error": f"Table '{table_name}' not found."}
    try:
        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            await con.execute(
                psycopg.sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                    psycopg.sql.Identifier(SCHEMA),
                    psycopg.sql.Identifier(table_name),
                )
            )
            await con.commit()
        del loaded_tables[table_name]
        return {"deleted": table_name, "remaining": list(loaded_tables.keys())}
    except Exception as e:
        return {"error": str(e)}


# ── Upload ───────────────────────────────────────────────────────────────────


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Parse uploaded CSV, create table in PostgreSQL, load data."""
    content = await file.read()
    text = content.decode("utf-8-sig")  # handle BOM
    reader = csv.reader(io.StringIO(text))

    rows_raw = list(reader)
    if len(rows_raw) < 2:
        return {"error": "CSV must have a header row and at least one data row."}

    raw_headers = rows_raw[0]
    data_rows = rows_raw[1:]

    # Sanitize column names (deduplicate if needed)
    col_names: list[str] = []
    seen: set[str] = set()
    for h in raw_headers:
        name = sanitize_name(h) or "col"
        base = name
        i = 2
        while name in seen:
            name = f"{base}_{i}"
            i += 1
        seen.add(name)
        col_names.append(name)

    # Infer types from data
    col_types: list[str] = []
    for ci in range(len(col_names)):
        sample = [r[ci] if ci < len(r) else "" for r in data_rows[:100]]
        col_types.append(infer_pg_type(sample, col_names[ci]))

    # Table name from file name — deduplicate if already loaded
    base_table = sanitize_name(
        os.path.splitext(file.filename or "upload")[0]
    ) or "csv_data"
    table_name = base_table
    suffix = 2
    while table_name in loaded_tables:
        table_name = f"{base_table}_{suffix}"
        suffix += 1

    try:
        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            # Create schema
            await con.execute(
                psycopg.sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    psycopg.sql.Identifier(SCHEMA)
                )
            )
            # Drop if exists (safety net)
            await con.execute(
                psycopg.sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                    psycopg.sql.Identifier(SCHEMA),
                    psycopg.sql.Identifier(table_name),
                )
            )
            # Create table
            col_defs = ", ".join(
                f"{psycopg.sql.Identifier(n).as_string(con)} {t}"
                for n, t in zip(col_names, col_types)
            )
            await con.execute(
                psycopg.sql.SQL(
                    "CREATE TABLE {}.{} (row_id serial PRIMARY KEY, {})"
                ).format(
                    psycopg.sql.Identifier(SCHEMA),
                    psycopg.sql.Identifier(table_name),
                    psycopg.sql.SQL(col_defs),
                )
            )
            # Insert data
            placeholders = ", ".join(["%s"] * len(col_names))
            col_ids = ", ".join(
                psycopg.sql.Identifier(n).as_string(con) for n in col_names
            )
            insert_sql = psycopg.sql.SQL(
                "INSERT INTO {}.{} (" + col_ids + ") VALUES (" + placeholders + ")"
            ).format(
                psycopg.sql.Identifier(SCHEMA),
                psycopg.sql.Identifier(table_name),
            )

            for row in data_rows:
                values: list[str | float | None] = []
                for ci, ct in enumerate(col_types):
                    val = row[ci].strip() if ci < len(row) else ""
                    if not val:
                        values.append(None)
                    elif ct in ("bigint", "double precision"):
                        try:
                            values.append(float(val.replace(",", "")))
                        except ValueError:
                            values.append(val)
                    else:
                        values.append(val)
                await con.execute(insert_sql, values)

            await con.commit()

            # Track the table
            loaded_tables[table_name] = {
                "filename": file.filename or "upload.csv",
                "columns": col_names,
                "types": col_types,
                "rows": len(data_rows),
            }

            return {
                "table": f"{SCHEMA}.{table_name}",
                "columns": col_names,
                "types": col_types,
                "rows_loaded": len(data_rows),
            }
    except Exception as e:
        return {"error": str(e)}


# ── Query (SQL-RAG) ─────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    table: str | None = None


async def get_table_context(
    con: psycopg.AsyncConnection, table: str
) -> tuple[str, str, int]:
    """Return (schema_desc, sample_text, total_rows) for a table."""
    schema_rows = await (
        await con.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (SCHEMA, table),
        )
    ).fetchall()

    schema_desc = "\n".join(
        f"  - {r[0]} ({r[1]})" for r in schema_rows if r[0] != "row_id"
    )

    sample = await (
        await con.execute(
            psycopg.sql.SQL("SELECT * FROM {}.{} LIMIT 5").format(
                psycopg.sql.Identifier(SCHEMA),
                psycopg.sql.Identifier(table),
            )
        )
    ).fetchall()

    col_names = [
        d[0]
        for d in (
            await con.execute(
                psycopg.sql.SQL("SELECT * FROM {}.{} LIMIT 0").format(
                    psycopg.sql.Identifier(SCHEMA),
                    psycopg.sql.Identifier(table),
                )
            )
        ).description
        or []
    ]

    sample_text = ""
    for row in sample:
        sample_text += (
            " | ".join(
                f"{col_names[i]}={row[i]}"
                for i in range(len(row))
                if col_names[i] != "row_id"
            )
            + "\n"
        )

    count_row = await (
        await con.execute(
            psycopg.sql.SQL("SELECT count(*) FROM {}.{}").format(
                psycopg.sql.Identifier(SCHEMA),
                psycopg.sql.Identifier(table),
            )
        )
    ).fetchone()
    total_rows = count_row[0] if count_row else 0

    return schema_desc, sample_text, total_rows


async def llm_call(system: str, user: str) -> str:
    """Call OpenAI chat completion and return the response text."""
    response = await oai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def extract_sql(text: str) -> str | None:
    """Extract a SQL query from LLM response text."""
    import re as _re

    # Try ```sql ... ``` block first
    m = _re.search(r"```(?:sql)?\s*\n?(.*?)```", text, _re.DOTALL | _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try standalone SELECT/WITH statement
    m = _re.search(
        r"((?:SELECT|WITH)\s+.+?)(?:\n\n|$)", text, _re.DOTALL | _re.IGNORECASE
    )
    if m:
        return m.group(1).strip().rstrip(";") + ";"
    return None


@app.post("/query")
async def query_data(req: QueryRequest):
    """Two-step SQL-RAG: LLM generates SQL → execute → LLM formats answer."""
    table = req.table
    if not table:
        # Default to the most recently loaded table
        if loaded_tables:
            table = list(loaded_tables.keys())[-1]
        else:
            return {"error": "No table loaded. Upload a CSV first."}

    try:
        async with await psycopg.AsyncConnection.connect(DB_URL) as con:
            schema_desc, sample_text, total_rows = await get_table_context(
                con, table
            )

            if not schema_desc:
                return {"error": f"Table {SCHEMA}.{table} not found."}

            full_table = f'"{SCHEMA}"."{table}"'

            # Build context for ALL loaded tables so LLM can JOIN or pick
            all_tables_context = ""
            if len(loaded_tables) > 1:
                for tname in loaded_tables:
                    t_schema, t_sample, t_rows = await get_table_context(
                        con, tname
                    )
                    t_full = f'"{SCHEMA}"."{tname}"'
                    all_tables_context += (
                        f"\nTable: {t_full} ({t_rows} rows)\n"
                        f"Columns:\n{t_schema}\n"
                        f"Sample data:\n{t_sample}\n"
                    )
            else:
                all_tables_context = (
                    f"\nTable: {full_table} ({total_rows} rows)\n"
                    f"Columns:\n{schema_desc}\n"
                    f"Sample data:\n{sample_text}\n"
                )

            # ── Step 1: Ask LLM to generate a SQL query ──
            gen_sql_prompt = (
                f"You are a PostgreSQL expert. "
                f"The following tables are available in the database:\n"
                f"{all_tables_context}\n"
                f"Write a single SQL SELECT query to answer the user's question. "
                f"Always use the full table path (e.g. \"{SCHEMA}\".\"table_name\"). "
                f"You may JOIN multiple tables if the question requires data from "
                f"more than one table. "
                f"Return ONLY the SQL query inside a ```sql code block. "
                f"Use aggregation functions (SUM, AVG, COUNT, MIN, MAX) when "
                f"the question asks for totals, averages, counts, etc. "
                f"Do NOT just SELECT rows — compute the answer in SQL."
            )

            sql_response = await llm_call(gen_sql_prompt, req.question)
            generated_sql = extract_sql(sql_response)

            query_result_text = ""
            query_error = ""
            executed_sql = ""

            async def try_execute(sql: str) -> tuple[str, str]:
                """Try to execute SQL. Returns (result_text, error)."""
                try:
                    await con.execute("SAVEPOINT query_exec")
                    await con.execute(
                        "SET LOCAL statement_timeout = '10s'"
                    )
                    cur = await con.execute(sql)
                    rows = await cur.fetchall()
                    col_names = [d[0] for d in cur.description or []]

                    if rows:
                        header = " | ".join(col_names)
                        separator = " | ".join(
                            "-" * len(c) for c in col_names
                        )
                        data_lines = []
                        for row in rows[:50]:
                            data_lines.append(
                                " | ".join(str(v) for v in row)
                            )
                        result = (
                            f"{header}\n{separator}\n"
                            + "\n".join(data_lines)
                        )
                        if len(rows) > 50:
                            result += f"\n... ({len(rows)} total rows)"
                    else:
                        result = "(no rows returned)"
                    await con.execute("RELEASE SAVEPOINT query_exec")
                    return result, ""
                except Exception as e:
                    try:
                        await con.execute(
                            "ROLLBACK TO SAVEPOINT query_exec"
                        )
                    except Exception:
                        pass
                    return "", str(e)

            if generated_sql:
                # ── Step 2: Execute the SQL query ──
                executed_sql = generated_sql
                query_result_text, query_error = await try_execute(
                    generated_sql
                )

                # ── Step 2b: Retry — if it failed, ask LLM to fix it ──
                if query_error and not query_result_text:
                    fix_prompt = (
                        f"You are a PostgreSQL expert. "
                        f"The following SQL query failed:\n"
                        f"```sql\n{generated_sql}\n```\n\n"
                        f"Error: {query_error}\n\n"
                        f"Available tables:\n{all_tables_context}\n"
                        f"Fix the query. Pay attention to column types — "
                        f"use explicit casts (e.g. column::bigint) when "
                        f"joining columns of different types. "
                        f"Return ONLY the corrected SQL inside a "
                        f"```sql code block."
                    )
                    fix_response = await llm_call(
                        fix_prompt, req.question
                    )
                    fixed_sql = extract_sql(fix_response)
                    if fixed_sql:
                        executed_sql = fixed_sql
                        query_result_text, query_error = (
                            await try_execute(fixed_sql)
                        )

            # ── Step 3: Ask LLM to format the final answer ──
            tables_summary = ", ".join(
                f'"{SCHEMA}"."{t}"' for t in loaded_tables
            ) or full_table

            if query_result_text:
                answer_prompt = (
                    f"You are a data analyst. The user asked a question about "
                    f"data in these tables: {tables_summary}.\n\n"
                    f"The following SQL was executed:\n```sql\n{executed_sql}\n```\n\n"
                    f"Query results:\n```\n{query_result_text}\n```\n\n"
                    f"Provide a clear, concise answer to the user's question "
                    f"based on these actual query results. "
                    f"Include the key numbers. "
                    f"If the result is a single value, state it directly."
                )
            elif query_error:
                answer_prompt = (
                    f"You are a data analyst. The user asked a question about "
                    f"data in these tables: {tables_summary}.\n\n"
                    f"Available tables:\n{all_tables_context}\n"
                    f"A SQL query was attempted but failed: {query_error}\n\n"
                    f"Try to answer the question as best you can from the "
                    f"table schemas and sample data, and suggest a corrected query."
                )
            else:
                answer_prompt = (
                    f"You are a data analyst. "
                    f"Available tables:\n{all_tables_context}\n"
                    f"Answer the user's question. If a SQL query would help, "
                    f"provide it using full table paths like \"{SCHEMA}\".\"table_name\"."
                )

            answer = await llm_call(answer_prompt, req.question)

            return {
                "answer": answer,
                "sql": executed_sql or None,
                "model": LLM_MODEL,
                "table": f"{SCHEMA}.{table}",
                "total_rows": total_rows,
            }
    except Exception as e:
        return {"error": str(e)}


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"  Database:  {DB_URL}")
    print(f"  LLM Model: {LLM_MODEL}")
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "(default: api.openai.com)")
    print(f"  API Key:   {'set (' + api_key[:8] + '...)' if api_key else 'NOT SET — export OPENAI_API_KEY first!'}")
    print(f"  Base URL:  {base_url}")
    print()
    print(f"  Open in browser: http://localhost:8765")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8765)
