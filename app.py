import streamlit as st
import cv2
import numpy as np
from socaity_face2face import Face2Face  # Updated import
import tempfile
import os

# --- Initialize Face2Face only once ---
@st.cache_resource
def init_f2f():
    return Face2Face(device_id=-1)  # Use CPU to avoid CUDA errors

f2f = init_f2f()

st.title("Face Swap App 🤖")
st.markdown("Upload a **source face** and a **target face**, then click 'Swap Faces' to get the swapped result.")

# --- File upload ---
source_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
target_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if source_file and target_file:
    # Display input images
    st.subheader("Uploaded Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(source_file, caption="Source", use_container_width=True)
    with col2:
        st.image(target_file, caption="Target", use_container_width=True)

    # --- Save to temp files ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as src_tmp:
        src_tmp.write(source_file.read())
        source_path = src_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tgt_tmp:
        tgt_tmp.write(target_file.read())
        target_path = tgt_tmp.name

    # --- Swap Faces button ---
    if st.button("Swap Faces"):
        st.subheader("Swapping Faces...")
        with st.spinner("Processing face swap, please wait..."):
            try:
                # --- Perform face swap ---
                swapped_img = f2f.swap_img_to_img(source_path, target_path)
                swapped_img_bgr = cv2.cvtColor(swapped_img, cv2.COLOR_RGB2BGR)

                # --- Display result ---
                st.image(swapped_img_bgr, caption="Swapped Face Result", use_container_width=True)

                # --- Download button ---
                result_path = "swapped_result.jpg"
                cv2.imwrite(result_path, swapped_img_bgr)
                with open(result_path, "rb") as f:
                    st.download_button("Download Swapped Image", f, file_name="swapped_result.jpg")

            except Exception as e:
                st.error(f"Face swapping failed: {e}")
            finally:
                # --- Clean up result file ---
                if os.path.exists(result_path):
                    os.remove(result_path)

    # --- Clean up temp files ---
    try:
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(target_path):
            os.remove(target_path)
    except Exception as e:
        st.warning(f"Failed to clean up temporary files: {e}")
else:
    st.info("Please upload both a source and target image to proceed.")
