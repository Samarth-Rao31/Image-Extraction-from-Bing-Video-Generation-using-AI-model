import streamlit as st
from bing_image_downloader import downloader
import os
import pytesseract
from PIL import Image
import io
from rembg import remove as LowRemove
from backgroundremover.bg import remove as HighRemove
import paddleocr
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\464J0196\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Users\464J0196\AppData\Local\Programs\Tesseract-OCR\tessdata'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

OCRlanguages = {
    "Abaza": "abq",
    "Adyghe": "ady",
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "amh",
    "Ancient Greek": "grc",
    "Angika": "ang",
    "Arabic": "ar",
    "Armenian": "hye",
    "Assamese": "asm",
    "Avar": "ava",
    "Azerbaijani": "az",
    "Azerbaijani (Cyrillic)": "aze_cyrl",
    "Basque": "eus",
    "Belarusian": "be",
    "Bengali": "ben",
    "Bhojpuri": "bho",
    "Bihari": "bh",
    "Bosnian": "bs",
    "Breton": "bre",
    "Bulgarian": "bg",
    "Burmese": "mya",
    "Catalan": "cat",
    "Cebuano": "ceb",
    "Cherokee": "chr",
    "Chinese and English": "ch",
    "Chinese Simplified": "chi_sim",
    "Chinese Simplified Vertical": "chi_sim_vert",
    "Chinese Traditional": "ch_tra",
    "Chinese Traditional Vertical": "chi_tra_vert",
    "Corsican": "cos",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dargwa": "dar",
    "Divehi": "div",
    "Dutch": "nl",
    "Dzongkha": "dzo",
    "English": "en",
    "Esperanto": "epo",
    "Estonian": "et",
    "Faroese": "fao",
    "Filipino": "fil",
    "Finnish": "fin",
    "French": "fr",
    "French (Old)": "frm",
    "Frisian": "fry",
    "Galician": "glg",
    "Georgian": "kat",
    "Georgian (Old)": "kat_old",
    "German": "german",
    "German (Fraktur)": "deu_latf",
    "Goan Konkani": "gom",
    "Greek": "ell",
    "Gujarati": "guj",
    "Haitian Creole": "hat",
    "Hebrew": "heb",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Ingush": "inh",
    "Inuktitut": "iku",
    "Irish": "ga",
    "Italian": "it",
    "Italian (Old)": "ita_old",
    "Japanese": "japan",
    "Japanese Vertical": "jpn_vert",
    "Javanese": "jav",
    "Kabardian": "kbd",
    "Kannada": "kan",
    "Kazakh": "kaz",
    "Khmer": "khm",
    "Kirghiz": "kir",
    "Korean": "korean",
    "Kurdish": "ku",
    "Lak": "lbe",
    "Lao": "lao",
    "Latin": "lat",
    "Latvian": "lv",
    "Lezghian": "lez",
    "Lithuanian": "lt",
    "Luxembourgish": "ltz",
    "Macedonian": "mkd",
    "Magahi": "mah",
    "Maithili": "mai",
    "Malay": "ms",
    "Malayalam": "mal",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Math / equation detection mode": "equ",
    "Middle English": "enm",
    "Mongolian": "mn",
    "Nagpur": "sck",
    "Nepali": "ne",
    "Newari": "new",
    "Norwegian": "no",
    "Occitan": "oc",
    "Oriya": "ori",
    "Ossetic": "osd",
    "Pashto": "pus",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pan",
    "Quechua": "que",
    "Romanian": "ro",
    "Russian": "ru",
    "Sanskrit": "san",
    "Saudi Arabia": "sa",
    "Scottish Gaelic": "gla",
    "Serbian": "srp",
    "Serbian (Cyrillic)": "rs_cyrillic",
    "Serbian (Latin)": "rs_latin",
    "Sindhi": "sin",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Spanish (Old)": "spa_old",
    "Sundanese": "sun",
    "Swahili": "sw",
    "Swedish": "sv",
    "Syriac": "syr",
    "Tabassaran": "tab",
    "Tagalog": "tl",
    "Tajik": "tgk",
    "Tamil": "ta",
    "Tatar": "tat",
    "Telugu": "te",
    "Thai": "tha",
    "Tibetan": "bod",
    "Tigrinya": "tir",
    "Tongan": "ton",
    "Turkish": "tr",
    "Uighur": "uig",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Uzbek (Cyrillic)": "uzb_cyrl",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Yiddish": "yid",
    "Yoruba": "yor"

}


def download_images(query, limit, output_dir, retry_limit=3, adult_filter_off=True, force_replace=False, timeout=60):
    retries = 0
    while retries < retry_limit:
        try:
            downloader.download(query, limit=limit, output_dir=output_dir, adult_filter_off=adult_filter_off, force_replace=force_replace, timeout=timeout)
            return True
        except Exception as e:
            retries += 1
            st.warning(f"Download failed. Retrying... Attempt {retries}/{retry_limit}")
    st.error(f"Failed to download images for query: {query}")
    return False

def show_images_in_dir(output_dir):
    if not os.path.exists(output_dir):
        st.warning("No images found. Please download images first.")
        return
    
    image_files = os.listdir(output_dir)
    
    if not image_files:
        st.warning("No Images")
        return
    
    cols = st.columns(2)
    for idx, image_file in enumerate(image_files):
        col = cols[idx % 2]
        with col:
            st.image(os.path.join(output_dir, image_file))
            with open(os.path.join(output_dir, image_file), "rb") as file:
                st.download_button(
                    label=f"Download",
                    data=file,
                    file_name=image_file,
                    mime="image/jpeg"
                )

def extract_text(image, lang):
    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(image, lang=lang)
    return extracted_text

if __name__ == "__main__":
    st.title("Image Processing")
    app_mode = st.sidebar.selectbox("Choose the App Mode", [ "Search Images", "Remove Background","Text Extract"])

    if app_mode == "Remove Background":

        def remove_bg(data, model_name="u2net_human_seg"):
            img = HighRemove(data, model_name=model_name,
                            alpha_matting=True,
                            alpha_matting_foreground_threshold=240,
                            alpha_matting_background_threshold=10,
                            alpha_matting_erode_structure_size=10,
                            alpha_matting_base_size=1000)
            return img

        def classify_resolution(width, height, threshold_width=1920, threshold_height=1080):
            if width >= threshold_width and height >= threshold_height:
                return "High"
            else:
                return "Low"

        def get_image_resolution(image_path):
            with Image.open(image_path) as img:
                width, height = img.size
                return width, height

        def main(image_path):
            width, height = get_image_resolution(image_path)
            resolution_class = classify_resolution(width, height)
            st.write(f"Resolution: {width} x {height} - {resolution_class}")
            
            if resolution_class == "Low":
                input_img = Image.open(image_path)
                output_img = LowRemove(input_img)
            else:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                result_img = remove_bg(image_data)
                output_img = Image.open(io.BytesIO(result_img))
            
            st.image(output_img, caption='Processed Image', use_column_width=True)

        # Streamlit interface
        st.title('Background Removal')
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Save uploaded image locally
            image_path = "uploaded_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process and display the image
            main(image_path)
    
    elif app_mode == "Search Images":
        st.title("Image Search")
        output_dir = "dataset"
        
        query = st.text_input("Enter search query:")
        query = query.strip()
        limit = st.slider("Number of images:", min_value=1, max_value=50, value=1)
        
        if st.button("Search"):
            if download_images(query, limit, output_dir):
                query_output_dir = os.path.join(output_dir, query)
            st.write(query_output_dir)
            show_images_in_dir(query_output_dir)

    elif app_mode == "Text Extract":
        st.title('Text Extraction from Image')
    
        # File upload and language selection
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            selected_lang = st.selectbox('Select language', list(OCRlanguages.keys()))

            try:
                image = np.array(image)
                ocr_model = paddleocr.PaddleOCR(lang=OCRlanguages[selected_lang])
                result = ocr_model.ocr(image)

                recognized_text = ""
                for line in result:
                    for word in line:
                        recognized_text += word[1][0] + " "
                
                st.text_area("Recognized Text", value=recognized_text, height=200)

            except:
                image = Image.open(uploaded_image)
                extracted_text = pytesseract.image_to_string(image, lang=OCRlanguages[selected_lang])
                st.header('Extracted Text')
                st.write(extracted_text)
