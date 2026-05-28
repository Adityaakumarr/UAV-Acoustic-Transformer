import pypdf

reader = pypdf.PdfReader("Listening_To_UAV_3d_Trajectory_Estimation_Via_Acoustic_Transformer.pdf")
text = reader.pages[3].extract_text()
clean_text = text.encode('ascii', errors='ignore').decode('ascii')
print(clean_text)
