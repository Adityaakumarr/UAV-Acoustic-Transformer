import pypdf

reader = pypdf.PdfReader("Listening_To_UAV_3d_Trajectory_Estimation_Via_Acoustic_Transformer.pdf")

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    lines = text.split("\n")
    for line in lines:
        if "fig. 2" in line.lower() or "figure 2" in line.lower():
            # If it's not the caption itself
            if "comparison between predicted" not in line.lower():
                clean_line = line.strip().encode('ascii', errors='ignore').decode('ascii')
                print(f"Page {i+1} mentions: {clean_line}")
