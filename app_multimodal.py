"""
Multi-Modal Screenshot Search Application

This Streamlit app performs dual-vector search using:
1. Visual similarity (DINOv2)
2. Text similarity (Nomic + OCR)

Combined with weighted scoring to find the best match.
"""

import streamlit as st
from PIL import Image

from qdrant_client_helper import get_client, COLLECTION_NAME
from models.dinov2_model import get_visual_embedding
from models.nomic_model import get_text_embedding
from utils.ocr_extractor import extract_text_from_image


def search_collection(client, vector_name, vector, limit):
    """
    Run a named-vector search against Qdrant across client versions.
    """
    if hasattr(client, "search"):
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=(vector_name, vector),
            limit=limit
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using=vector_name,
        limit=limit,
        with_payload=True
    )
    return response.points


st.set_page_config(
    page_title="Multi-Modal Screenshot Search",
    page_icon="S",
    layout="wide"
)

st.title("Multi-Modal ERP Screenshot Search")
st.markdown("**DINOv2 Visual Search + Nomic Text Search + Tesseract OCR**")

st.sidebar.header("Search Configuration")

w1 = st.sidebar.slider(
    "Visual Weight (w1)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Weight for visual similarity score"
)

w2 = st.sidebar.slider(
    "Text Weight (w2)",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Weight for text similarity score"
)

total_weight = w1 + w2
if total_weight > 0:
    w1 = w1 / total_weight
    w2 = w2 / total_weight

st.sidebar.info(f"Normalized Weights:\n- Visual: {w1:.2f}\n- Text: {w2:.2f}")

top_k = st.sidebar.number_input(
    "Top-K Candidates",
    min_value=1,
    max_value=20,
    value=10,
    help="Number of candidates to retrieve from each search"
)

st.header("Upload Query Screenshot")
uploaded_file = st.file_uploader(
    "Choose a screenshot to search",
    type=["png", "jpg", "jpeg", "bmp", "gif", "webp"]
)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Query Image")
        st.image(
            query_image,
            caption=f"Size: {query_image.size[0]} x {query_image.size[1]}",
            use_container_width=True
        )

    with col2:
        st.subheader("Search Progress")

        with st.spinner("Extracting text from query..."):
            query_text = extract_text_from_image(query_image)

            if query_text:
                if len(query_text) > 100:
                    st.success(f'Extracted Text: "{query_text[:100]}..."')
                else:
                    st.success(f'Extracted Text: "{query_text}"')
            else:
                st.warning("No text found in query image")

        with st.spinner("Generating visual embedding..."):
            query_visual_emb = get_visual_embedding(query_image)
            st.success("Visual embedding generated (768-dim)")

        with st.spinner("Generating text embedding..."):
            query_text_emb = get_text_embedding(query_text)
            st.success("Text embedding generated (768-dim)")

    st.header("Search Results")

    with st.spinner("Searching database..."):
        client = get_client()

        visual_results = search_collection(
            client,
            "image_vector",
            query_visual_emb.tolist(),
            top_k
        )

        text_results = search_collection(
            client,
            "text_vector",
            query_text_emb.tolist(),
            top_k
        )

    st.subheader("Candidate Scoring")

    combined_scores = {}

    for result in visual_results:
        img_path = result.payload["image_path"]
        combined_scores[img_path] = {
            "visual_score": result.score,
            "text_score": 0.0,
            "payload": result.payload
        }

    for result in text_results:
        img_path = result.payload["image_path"]
        if img_path in combined_scores:
            combined_scores[img_path]["text_score"] = result.score
        else:
            combined_scores[img_path] = {
                "visual_score": 0.0,
                "text_score": result.score,
                "payload": result.payload
            }

    for img_path in combined_scores:
        v_score = combined_scores[img_path]["visual_score"]
        t_score = combined_scores[img_path]["text_score"]
        final_score = w1 * v_score + w2 * t_score
        combined_scores[img_path]["final_score"] = final_score

    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1]["final_score"],
        reverse=True
    )

    if sorted_results:
        st.success(f"Found {len(sorted_results)} candidates")

        best_match_path, best_match_data = sorted_results[0]

        st.subheader("Best Match")

        col1, col2 = st.columns([1, 1])

        with col1:
            try:
                result_image = Image.open(best_match_data["payload"]["image_path"])
                st.image(
                    result_image,
                    caption=f'Match: {best_match_data["payload"]["filename"]}',
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col2:
            st.markdown("### Scores")
            st.metric("Final Score", f'{best_match_data["final_score"]:.4f}')

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Visual Score", f'{best_match_data["visual_score"]:.4f}')
            with col_b:
                st.metric("Text Score", f'{best_match_data["text_score"]:.4f}')

            st.markdown("### Metadata")
            st.write(f'**Filename:** {best_match_data["payload"]["filename"]}')
            st.write(
                f'**Size:** {best_match_data["payload"]["image_width"]} x '
                f'{best_match_data["payload"]["image_height"]}'
            )

            if best_match_data["payload"]["extracted_text"]:
                with st.expander("Extracted Text"):
                    st.text(best_match_data["payload"]["extracted_text"])

        st.subheader("All Candidates")

        for i, (img_path, data) in enumerate(sorted_results[:10], 1):
            with st.expander(f'#{i}: {data["payload"]["filename"]} - Score: {data["final_score"]:.4f}'):
                col1, col2 = st.columns([1, 2])

                with col1:
                    try:
                        img = Image.open(data["payload"]["image_path"])
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.error("Cannot load image")

                with col2:
                    st.write(f'**Final Score:** {data["final_score"]:.4f}')
                    st.write(f'**Visual Score:** {data["visual_score"]:.4f}')
                    st.write(f'**Text Score:** {data["text_score"]:.4f}')
                    st.write(
                        f'**Size:** {data["payload"]["image_width"]} x '
                        f'{data["payload"]["image_height"]}'
                    )

                    if data["payload"]["extracted_text"]:
                        st.write(f'**Text:** {data["payload"]["extracted_text"][:200]}...')

    else:
        st.warning("No results found")

else:
    st.info("Upload a screenshot to search the database")
