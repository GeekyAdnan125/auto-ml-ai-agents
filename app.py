
import hashlib
import json
from pathlib import Path
import uuid
import traceback
import pandas as pd
import streamlit as st

from graph.workflow import run_automl_workflow
from rag.vector_store import index_reports
from rag.rag_chain import answer_question
from rag.csv_indexer import index_csv
from rag.csv_qa import answer_csv_question


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
REPORT_DIR = BASE_DIR / "reports"
SAMPLE_PATH = BASE_DIR / "data" / "sample.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AutoML Agent Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. LIGHT THEME + CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        color: #172033;
    }

    [data-testid="stHeader"] {
        background-color: rgba(247, 249, 252, 0.95);
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5eaf2;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid #e1e8f0;
        border-radius: 18px;
        background: linear-gradient(120deg, #ffffff, #edf4ff);
        margin-bottom: 1.3rem;
    }

    .hero h1 {
        font-size: 2rem;
        color: #172033;
        margin-bottom: 0.35rem;
    }

    .hero p {
        color: #5d6b82;
        margin-bottom: 0;
        font-size: 1rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf2;
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e5eaf2;
        border-radius: 10px;
        overflow: hidden;
    }

    .section-caption {
        color: #64748b;
        font-size: 0.92rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def initialize_state():
    defaults = {
        "csv_messages": [],
        "report_messages": [],
        "csv_indexed_id": None,
        "csv_indexed_chunks": 0,
        "automl_result": None,
        "automl_dataset_id": None,
        "automl_target": None,
        "automl_task": "auto",
        "report_indexed": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def read_csv_safely(file_bytes):
    """Read common CSV encodings without hiding parsing errors."""
    from io import BytesIO

    errors = []

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(
        "Could not parse the CSV. Check its format and delimiters. "
        + " | ".join(errors)
    )


def get_dataset_id(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def show_sources(sources, key_prefix):
    if not sources:
        return

    with st.expander("Retrieved sources", expanded=False):
        for i, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                filename = source.get("filename", "CSV")
                start = source.get("row_start")
                end = source.get("row_end")

                if start is not None and end is not None:
                    st.write(
                        f"{i}. **{filename}** — rows/lines "
                        f"{start}–{end}"
                    )
                else:
                    st.write(f"{i}. **{filename}**")
            else:
                st.write(f"{i}. {source}")


def show_chat_history(messages, key_prefix):
    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(
                message.get("sources", []),
                f"{key_prefix}_{i}",
            )


def render_model_results(result):
    if not result:
        st.info("Run the AutoML workflow to see model results.")
        return

    error = result.get("error")
    if error:
        st.error(f"Workflow error: {error}")
        return

    validation = result.get("validation_result", {})
    if validation:
        st.success(
            validation.get("message", "Validation completed.")
        )

    st.success("AutoML workflow completed.")

    model_result = result.get("model_result", {})
    models = model_result.get("models", {})

    if not models:
        st.warning("No model comparison results were returned.")
    else:
        rows = []

        for name, details in models.items():
            row = {
                "Model": name,
                "Status": details.get("status", "unknown"),
            }
            row.update(details.get("metrics", {}))
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)

        successful = [
            (name, details)
            for name, details in models.items()
            if details.get("status") == "success"
        ]

        if successful:
            chart_data = pd.DataFrame(
                {
                    name: details.get("metrics", {})
                    for name, details in successful
                }
            ).T

            numeric_columns = chart_data.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                selected_metrics = st.multiselect(
                    "Metrics to compare",
                    options=numeric_columns,
                    default=numeric_columns[:2],
                    key="automl_metrics_to_plot",
                )

                if selected_metrics:
                    st.bar_chart(chart_data[selected_metrics])

        best_model = model_result.get("best_model")
        selection_metric = model_result.get(
            "selection_metric",
            "Not reported",
        )

        col1, col2 = st.columns(2)
        col1.metric(
            "Selected model",
            str(best_model or "Not reported"),
        )
        col2.metric(
            "Selection metric",
            str(selection_metric),
        )

    st.subheader("AI explanation")
    answer = result.get("llm_answer")

    if answer:
        st.markdown(answer)
    else:
        st.info(
            "No AI explanation was returned. Check the "
            "configuration in agents/llm_explainer.py."
        )


initialize_state()


# =========================================================
# 4. HEADER + SIDEBAR
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 AutoML Agent Studio</h1>
        <p>
            Explore data, train machine-learning models, and ask
            questions about your datasets using local RAG.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Workspace")
    st.caption("Configure your dataset and experiment.")

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Upload a structured CSV file to get started.",
    )

    st.divider()

    task = st.selectbox(
        "Machine-learning task",
        ["auto", "regression", "classification"],
        help="Choose auto to let your existing workflow determine "
             "the task when supported.",
    )

    target_column = None

    if uploaded_file is not None:
        st.caption("The target column is selected in the AutoML tab.")

    user_question = st.text_area(
        "AI explanation instructions",
        value=(
            "Explain the selected model, compare its metrics, "
            "and suggest practical improvements."
        ),
        help="Instructions passed to your existing AutoML workflow.",
    )

    st.divider()
    st.caption("Local AI models: Gemma + Nomic embeddings")


# =========================================================
# 5. LOAD DATASET
# =========================================================

df = None
dataset_id = None
filename = None
load_error = None

if uploaded_file is not None:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        dataset_id = get_dataset_id(uploaded_bytes)
        filename = uploaded_file.name

        # Reload only when a genuinely different file is uploaded.
        if st.session_state.get("active_dataset_id") != dataset_id:
            df = read_csv_safely(uploaded_bytes)

            st.session_state["active_dataset_id"] = dataset_id
            st.session_state["active_dataset_df"] = df
            st.session_state["active_dataset_filename"] = filename

            # A different dataset gets a fresh chat.
            st.session_state["csv_messages"] = []
        else:
            df = st.session_state["active_dataset_df"]

        if df.empty:
            load_error = "The uploaded CSV has no data rows."
        elif not df.columns.is_unique:
            load_error = "The CSV contains duplicate column names."
        else:
            st.session_state["active_dataset_df"] = df

    except Exception as exc:
        load_error = str(exc)

elif SAMPLE_PATH.exists():
    try:
        sample_bytes = SAMPLE_PATH.read_bytes()
        sample_id = get_dataset_id(sample_bytes)
        df = read_csv_safely(sample_bytes)
        dataset_id = sample_id
        filename = SAMPLE_PATH.name

        st.info(
            "No file uploaded. The sample dataset is available "
            "for exploration; upload your own CSV to use it instead."
        )
    except Exception as exc:
        load_error = f"Could not load the sample CSV: {exc}"


if load_error:
    st.error(load_error)
    df = None
    dataset_id = None


# =========================================================
# 6. NAVIGATION
# =========================================================

overview_tab, automl_tab, csv_tab, reports_tab = st.tabs(
    [
        "Overview",
        "AutoML Workspace",
        "CSV AI Analyst",
        "Reports & Assistant",
    ]
)


# =========================================================
# 7. OVERVIEW
# =========================================================

with overview_tab:
    st.subheader("Dataset overview")

    if df is None:
        st.info(
            "Upload a valid CSV using the sidebar to view "
            "dataset statistics and previews."
        )
    else:
        missing_cells = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Missing cells", f"{missing_cells:,}")
        c4.metric("Duplicate rows", f"{duplicate_rows:,}")

        st.caption(f"Dataset: {filename}")

        st.markdown("#### Data preview")
        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=420,
        )

        with st.expander("Column details"):
            details = pd.DataFrame({
                "Column": df.columns.astype(str),
                "Data type": df.dtypes.astype(str).values,
                "Missing values": df.isna().sum().values,
                "Unique values": df.nunique(dropna=True).values,
            })
            st.dataframe(details, use_container_width=True)

        with st.expander("Quick numeric summary"):
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.empty:
                st.info("No numeric columns were detected.")
            else:
                st.dataframe(
                    numeric_df.describe().T,
                    use_container_width=True,
                )


# =========================================================
# 8. AUTOML WORKSPACE
# =========================================================

with automl_tab:
    st.subheader("Train and compare machine-learning models")
    st.caption(
        "This section uses your existing LangGraph AutoML workflow."
    )

    if df is None or dataset_id is None:
        st.info("Upload a valid CSV to configure an experiment.")
    else:
        st.write(
            f"**Dataset:** {filename} · "
            f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns"
        )

        selected_target = st.selectbox(
            "Target column",
            options=list(df.columns),
            index=len(df.columns) - 1,
            key=f"target_{dataset_id}",
        )

        st.caption(
            "Choose the value or category that the model should predict."
        )

        if st.button(
            "Run AutoML workflow",
            type="primary",
            key="run_automl",
        ):
            dataset_path = (
                UPLOAD_DIR / f"dataset_{uuid.uuid4().hex}.csv"
            )
            df.to_csv(dataset_path, index=False)

            try:
                with st.spinner(
                    "Validating data, engineering features, "
                    "training models, and generating an explanation..."
                ):
                    result = run_automl_workflow(
                        dataset_path=str(dataset_path),
                        target_column=selected_target,
                        task=task,
                        user_question=user_question,
                    )

                st.session_state["automl_result"] = result
                st.session_state["automl_dataset_id"] = dataset_id
                st.session_state["automl_target"] = selected_target
                st.session_state["automl_task"] = task

            except Exception as exc:
                st.error(f"AutoML workflow failed: {exc}")

        # Do not show results for an unrelated previously uploaded file.
        if (
            st.session_state.get("automl_dataset_id") == dataset_id
            and st.session_state.get("automl_target") is not None
        ):
            render_model_results(
                st.session_state.get("automl_result")
            )
        else:
            st.info(
                "Configure the target column and run the workflow "
                "to see model metrics and explanations."
            )


# =========================================================
# 9. CSV RAG ASSISTANT
# =========================================================

with csv_tab:
    st.subheader("Ask questions about your uploaded dataset")
    st.caption(
        "Semantic search retrieves relevant CSV chunks. "
        "Answers are generated locally using Gemma via Ollama."
    )

    if df is None or dataset_id is None:
        st.info("Upload a valid CSV before indexing it.")
    else:
        st.write(f"**Active dataset:** {filename}")

        indexed = (
            st.session_state.get("csv_indexed_id") == dataset_id
        )

        if indexed:
            st.success(
                f"CSV index ready · "
                f"{st.session_state.get('csv_indexed_chunks', 0)} chunks"
            )
        else:
            st.warning("This dataset has not been indexed yet.")

        if st.button(
            "Build / Refresh CSV index",
            type="primary",
            key="index_csv_button",
        ):
            try:
                with st.spinner(
                    "Converting CSV rows, generating embeddings, "
                    "and storing chunks in ChromaDB..."
                ):
                    count = index_csv(
                        df=df,
                        dataset_id=dataset_id,
                        filename=filename,
                    )

                st.session_state["csv_indexed_id"] = dataset_id
                st.session_state["csv_indexed_chunks"] = count
                st.success(f"Indexed {count} text chunks.")
                st.rerun()

            
            except Exception as exc:
                st.error(f"CSV indexing failed: {exc}")


        st.markdown("#### Conversation")

        show_chat_history(
            st.session_state["csv_messages"],
            "csv_history",
        )

        with st.form("csv_question_form", clear_on_submit=True):
            csv_question = st.text_input(
                "Your question",
                placeholder=(
                    "e.g. Explain the records for a specific customer..."
                ),
            )
            submit_csv_question = st.form_submit_button(
                "Ask about CSV",
                type="primary",
            )

        if submit_csv_question and csv_question.strip():
            if st.session_state.get("csv_indexed_id") != dataset_id:
                st.warning(
                    "Build the index for this dataset before asking questions."
                )
            else:
                st.session_state["csv_messages"].append({
                    "role": "user",
                    "content": csv_question.strip(),
                })

                try:
                    with st.spinner(
                        "Searching relevant chunks and generating an answer..."
                    ):
                        answer_result = answer_csv_question(
                            question=csv_question.strip(),
                            dataset_id=dataset_id,
                        )

                    st.session_state["csv_messages"].append({
                        "role": "assistant",
                        "content": answer_result.get(
                            "answer",
                            "No answer was returned.",
                        ),
                        "sources": answer_result.get("sources", []),
                    })

                    st.rerun()

                except Exception as exc:
                    st.error(f"CSV question failed: {exc}")

        st.caption(
            "Tip: semantic RAG is best for retrieving and explaining "
            "relevant records. Exact totals, global rankings, and "
            "statistical tests need full-dataset calculations."
        )


# =========================================================
# 10. REPORTS + REPORT-BASED RAG
# =========================================================

with reports_tab:
    st.subheader("Generated AutoML reports")

    report_files = [
        "analysis_report.json",
        "analysis_report.txt",
        "model_comparison.json",
    ]

    found_reports = False

    for report_name in report_files:
        report_path = REPORT_DIR / report_name

        if not report_path.exists():
            continue

        found_reports = True

        with st.expander(report_name):
            try:
                if report_path.suffix.lower() == ".json":
                    report_data = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    st.json(report_data)

                    download_data = json.dumps(
                        report_data,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                else:
                    download_data = report_path.read_text(
                        encoding="utf-8"
                    )
                    st.text(download_data)

                st.download_button(
                    f"Download {report_name}",
                    data=download_data,
                    file_name=report_name,
                    mime=(
                        "application/json"
                        if report_path.suffix.lower() == ".json"
                        else "text/plain"
                    ),
                    key=f"download_{report_name}",
                )

            except (OSError, json.JSONDecodeError) as exc:
                st.warning(f"Could not read {report_name}: {exc}")

    if not found_reports:
        st.info(
            "No reports found yet. Run the AutoML workflow first."
        )

    st.divider()
    st.subheader("Ask about AutoML reports")
    st.caption(
        "This assistant searches your generated analysis and "
        "model-comparison reports, not the uploaded CSV row index."
    )

    if st.button(
        "Index / Refresh AutoML reports",
        key="index_reports_button",
    ):
        try:
            with st.spinner("Indexing AutoML reports..."):
                count = index_reports()

            st.session_state["report_indexed"] = True
            st.success(f"Indexed {count} report chunks.")

        except Exception as exc:
            st.error(f"Report indexing failed: {exc}")

    show_chat_history(
        st.session_state["report_messages"],
        "report_history",
    )

    with st.form("report_question_form", clear_on_submit=True):
        report_question = st.text_input(
            "Question about reports",
            placeholder=(
                "e.g. Compare the reported model evaluation metrics..."
            ),
        )
        submit_report_question = st.form_submit_button(
            "Ask about reports",
            type="primary",
        )

    if submit_report_question and report_question.strip():
        st.session_state["report_messages"].append({
            "role": "user",
            "content": report_question.strip(),
        })

        try:
            with st.spinner("Searching AutoML reports..."):
                report_result = answer_question(
                    report_question.strip()
                )

            st.session_state["report_messages"].append({
                "role": "assistant",
                "content": report_result.get(
                    "answer",
                    "No answer was returned.",
                ),
                "sources": report_result.get("sources", []),
            })

            st.rerun()

        except Exception as exc:
            st.error(f"Report question failed: {exc}")


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()
st.caption(
    "AutoML Agent Studio · Streamlit · LangGraph · "
    "ChromaDB · Ollama"
)
