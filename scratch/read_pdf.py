import pypdf

reader = pypdf.PdfReader("Listening_To_UAV_3d_Trajectory_Estimation_Via_Acoustic_Transformer.pdf")
print(f"Total pages: {len(reader.pages)}")

keywords = ["blue", "red", "ground truth", "estimation", "prediction", "color", "dotted", "dashed", "Pham4"]

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    for kw in keywords:
        if kw.lower() in text.lower():
            print(f"Page {i+1} matches keyword '{kw}':")
            # print surrounding text of the matches
            lines = text.split("\n")
            for line in lines:
                if kw.lower() in line.lower():
                    clean_line = line.strip().encode('ascii', errors='ignore').decode('ascii')
                    print(f"  {clean_line}")
            print("-" * 50)
