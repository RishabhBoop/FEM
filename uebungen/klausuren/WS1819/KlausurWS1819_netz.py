import sys
sys.path.append(r"C:/Users/weju0001/Data/Python/MESH")
import numpy as np
import helper_funcs.meshtools as mt
import matplotlib.pyplot as plt


## Definition des Netzes, Klausur WW1819 ##

# 
NL=4
Ra=25e-2
HH=10e-2
Dx=3e-3
V0=1.

# Abstand platte 1 innen bis Platte NL innen
Breite=(2*NL-3)*Dx+2*1e-5
#Frequenz
f0=10e3
#Fuellstandshoehe
hh=HH/2.

# Dielektrizitaetszahlen
epsr=11
eps0= 8.854187817e-12
#Leitfaehigkeiten
sigma=1e-4
sigma0=1e-12

# kappa, Abkuerzung 
#kappa=sigma+2.*np.pi*f0*eps0*epsr*1J
#kappa0=2.*np.pi*f0*eps0*1J
#kappa=eps0*epsr
#kappa0=eps0

length=0.0053*np.sqrt(NL)

# aeusserer Kreis
p,v=mt.CircleSegments([0,0],Ra,edge_length=5*length)

# innere Rechtecke
RU=[]
RO=[]
HolPoi=[]
x0=-NL*Dx+Dx/2.
for k in range(NL):
  x=x0+k*2*Dx
  RU+=[[x,-HH/2.]]
  RO+=[[x+Dx,HH/2.]]
  # Rechteck mit eckigen Kanten
  p0,v0=mt.RectangleSegments(RU[-1],RO[-1],edge_lengthx=length/55,edge_lengthy=length/15)
  # Rechteck mit runden Kanten 
  #p0,v0=mt.ORecSegments(RU[-1],RO[-1],Dx/8,edge_lengthx=length/50,edge_lengthy=length/12,num_pc=15)
  p,v=mt.AddCurves(p,v,p0,v0)
  HolPoi+=[(x+Dx/2.,0)]


#Verfeinerung
def myrefine(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  [x0,y0]=np.array([center_tri[0],center_tri[1]])
  rsp=x0**2+y0**2
  if np.abs(x0)<1.15*NL*Dx and np.abs(y0)<1.15*HH/2:
    max_area=length*length/200
  elif np.abs(x0)<1.25*NL*Dx and np.abs(y0)<1.25*HH/2:
    max_area=length*length/20
  else:
    max_area=10*length*length
  return bool(area>max_area)

# Netz wird generiert
poi,tri,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,tri_refine=myrefine,holes=HolPoi, writeTo="mesh_WS1819.npz")

print("Anzahl der Punkte:           ",len(poi))
print("Anzahl der Dreieckselemente: ",len(tri))
# plt.show()


# Randkurven
Ps=[[Ra,0],[Ra,0]]
typ=['Segments']+NL*['Nodes']
for k in range(NL):
  Ps+=[RU[k],RU[k]]

bseg=mt.RetrieveSegments(poi,BouE,li_BE,Ps,typ)

# Boundary-Plot
for k in range(0,NL+1):
  mt.PlotBoundary(poi,bseg[k],typ[k])
# plt.show()
 
# Robin-Rand 
R0=bseg[0]

#Dirichlet-Rand
G0=[]
for k in range(1,NL+1):
  G0+=bseg[k] 

