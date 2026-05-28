import pypdf
import os

pdf_path = "Listening_To_UAV_3d_Trajectory_Estimation_Via_Acoustic_Transformer.pdf"
reader = pypdf.PdfReader(pdf_path)

out_dir = "outputs/pdf_images"
os.makedirs(out_dir, exist_ok=True)

img_count = 0
for page_num, page in enumerate(reader.pages):
    for image_file_object in page.images:
        img_count += 1
        img_name = f"page_{page_num+1}_img_{img_count}_{image_file_object.name}"
        img_path = os.path.join(out_dir, img_name)
        with open(img_path, "wb") as fp:
            fp.write(image_file_object.data)
        print(f"Extracted image to: {img_path}")

print(f"Extraction complete. Total images extracted: {img_count}")
