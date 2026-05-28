import pypdf

reader = pypdf.PdfReader("Listening_To_UAV_3d_Trajectory_Estimation_Via_Acoustic_Transformer.pdf")
text = reader.pages[3].extract_text()
lines = text.split("\n")

# Find index of line containing "Fig. 2"
fig2_idx = -1
for idx, line in enumerate(lines):
    if "fig. 2" in line.lower():
        fig2_idx = idx
        break

if fig2_idx != -1:
    print("Context around Fig. 2 in Page 4:")
    start = max(0, fig2_idx - 15)
    end = min(len(lines), fig2_idx + 15)
    for idx in range(start, end):
        clean_line = lines[idx].strip().encode('ascii', errors='ignore').decode('ascii')
        print(f"{idx}: {clean_line}")
else:
    print("Could not find Fig. 2 context")
