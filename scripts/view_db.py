import sqlite3

conn = sqlite3.connect("detections.db")
c = conn.cursor()
c.execute("SELECT * FROM detections ORDER BY id DESC")
rows = c.fetchall()
conn.close()

print(f"\n{'='*120}")
print(f"{'ID':<4}{'Object':<12}{'Conf%':<7}{'Colors':<22}{'Size cm':<11}{'Grasp':<11}{'Grip':<6}{'X,Y':<12}{'Text':<15}{'Time'}")
print(f"{'='*120}")
for r in rows:
    size = f"{r[4]}x{r[5]}" if r[4] else "N/A"
    xy = f"({r[9]},{r[10]})" if r[9] is not None else "N/A"
    print(f"{r[0]:<4}{r[1]:<12}{r[2]:<7}{str(r[3])[:21]:<22}{size:<11}{str(r[7]):<11}{str(r[8]):<6}{xy:<12}{str(r[6])[:14]:<15}{r[11]}")
print(f"{'='*120}")
print(f"Total: {len(rows)} detections\n")
