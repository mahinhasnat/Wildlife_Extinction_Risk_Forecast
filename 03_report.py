import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
B="/mnt/user-data/outputs/bd_wildlife"
s=getSampleStyleSheet(); S=[]
def h(t): S.append(Paragraph(t,s["Heading2"]))
def p(t): S.append(Paragraph(t,s["Normal"])); S.append(Spacer(1,6))
def im(f,c):
    S.append(Image(f"{B}/plots/{f}",width=400,height=400*0.58)); S.append(Paragraph(c,s["Italic"])); S.append(Spacer(1,8))
S.append(Paragraph("Bangladesh Wildlife: Extinction Status and 2030 Risk Projection",s["Title"]))
h("1. What is verified and what is not")
p("Bangladesh has two formal Red Lists (IUCN 2000 and 2015). No public species-level yearly series exists for 2010-2026; the new Red List began in 2026. "
  "Therefore the 2030 numbers below are a <b>model-based scenario</b> anchored on 2015 verified counts, not observed data. They become testable when the new Red List is published.")
h("2. Verified baseline (2015)")
p("1,619 species assessed. 31 regionally extinct (11 mammals, 19 birds, 1 reptile). 56 Critically Endangered, 181 Endangered, 153 Vulnerable (390 threatened). "
  "Mammals have the worst Red List Index (0.55), then butterflies (0.64).")
im("01_assessed_vs_threatened.png","Fig 1. Assessed vs threatened species by group.")
im("02_pct_threatened.png","Fig 2. Share threatened: butterflies 61.6%, mammals 27.5%, reptiles 22.8%.")
im("03_extinct_2000_vs_2015.png","Fig 3. Extinct lists 2000 vs 2015. Re-assessment, not only new losses.")
im("05_rli_2015.png","Fig 4. Red List Index 2015.")
S.append(PageBreak())
h("3. Method (ML/statistical projection)")
p("Monte Carlo simulation (20,000 runs). Each CR/EN/VU species gets an extinction hazard derived from IUCN Criterion E thresholds "
  "(CR 50% in 10 y, EN 20% in 20 y, VU 10% in 100 y). Because real extinctions are slower than thresholds, a realised fraction r ~ Beta(2,5) scales the hazard "
  "(<b>Assumption A2</b>). Species are independent (A3). For groups without published CR/EN/VU split, the national remainder is allocated proportionally (A1).")
pr=pd.read_csv(f"{B}/data/projection_new_extinctions.csv")
t=[["Group","Year","Expected new","5%","95%"]]+[[r.group,int(r.year),r[2],r.p05,r.p95] for r in pr.itertuples(index=False) if r.year==2030]
tb=Table(t); tb.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.4,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])); S.append(tb); S.append(Spacer(1,8))
im("06_projection_trajectory.png","Fig 5. Cumulative new regional extinctions since 2015 (scenario).")
im("07_projection_by_group_2030.png","Fig 6. Projected losses by group, 2030.")
im("08_sensitivity.png","Fig 7. Result depends strongly on r: 4.5 (r=0.05) to 66.5 (r=1.0) species.")
im("09_rli_2015_vs_2030.png","Fig 8. RLI lower-bound projection (extinctions only).")
h("4. Forestry")
p("Verified Global Forest Watch points: 261,746 ha tree cover lost 2001-2024 (13.5% of 2000 extent); 2017 peak ~28,328 ha; 2024 15,911 ha. "
  "Only 8 annual points were verifiable offline, so the trend is indicative (2030 central ~22.7k ha, 90% range 17.2k-32.4k).")
im("10_forest_tree_cover_loss.png","Fig 9. Verified tree-cover-loss points and indicative trend.")
h("5. Limits")
p("No species-level names are predicted as 'will go extinct'. The 2015 national lists should be requested from Forest Dept/IUCN for the species-level ML script (code/02). "
  "Tiger: 125 individuals reported for 2024 (parliament reply). Treat all 2030 outputs as scenarios with wide uncertainty.")
SimpleDocTemplate(f"{B}/Bangladesh_Wildlife_Report.pdf",pagesize=A4).build(S)
