# -*- coding: utf-8 -*-
"""
Klausur SS20
Methode der finiten Elemente in 2D
Netz

"""

import numpy as np
import meshtools as mt
import matplotlib.pyplot as plt


# Abmessungen
Ra=2
dd=0.0005
Rp=0.2
DD=0.005
epsilon_0=8.85418782e-12


##################################################################
##    Geometrie
##################################################################




length=0.01
epsL=length/100

# Definition der Punkte
P1=[0,Ra]
P2=[0,DD/2.+dd]
P3=[Rp-dd/2.,DD/2.+dd]
allt=np.linspace(np.pi/2.,-np.pi/2,int (200*dd/length) )
PCend1=[[Rp-dd/2.+dd/2*np.cos(tt),(DD+dd)/2.+dd/2*np.sin(tt)] for tt in allt]
PCend2=[[Rp-dd/2.+dd/2*np.cos(tt),-(DD+dd)/2.+dd/2*np.sin(tt)] for tt in allt]
P4=[Rp-dd/2,DD/2.]
P5=[0,DD/2.]
P6=[0,-DD/2.]
P7=[Rp-dd/2.,-DD/2.]
P8=[Rp-dd/2.,-DD/2.-dd]
P9=[0,-DD/2.-dd]
P10=[0,-Ra]

# Halbkreis und gerades Stück
p1,v1=mt.CircleSegments([0,0],Ra,a_min=-np.pi/2,a_max=np.pi/2,edge_length=50*length)
p2,v2=mt.PointSegments([P1,P2],edge_length=10*length)
p,v=mt.AddSegments(p1,p2)



# Abrundungen und gerade Stücke
# 1
if Rp>dd/2:
  p3,v3=mt.PointSegments([P2]+PCend1+[P5],edge_length=length/30.)
else:
  p3,v3=mt.PointSegments(PCend1,edge_length=length/30.)
p,v=mt.AddSegments(p,p3)
p4,v4=mt.PointSegments([P5,P6],edge_length=length/10.)
p,v=mt.AddSegments(p,p4)
# 2
if Rp>dd/2:
  p5,v5=mt.PointSegments([P6]+PCend2+[P9],edge_length=length/30.)
else:
  p5,v5=mt.PointSegments(PCend2,edge_length=length/30.)
p,v=mt.AddSegments(p,p5)
p6,v6=mt.PointSegments([P9,P10],edge_length=10*length)
p,v=mt.AddSegments(p,p6,closed=True)    


#Verfeinerung
def myrefine(tri_points, area):
  center_tri = np.sum(np.array(tri_points), axis=0)/3.
  r=center_tri[0]
  z=center_tri[1]
  rho_sq=r**2+z**2  
  
  if Rp<dd: 
    if rho_sq < 2*(DD**2/4.+dd**2+DD*dd):
      max_area=0.5*(length/10.)**2
    else:
      max_area=100*length**2*rho_sq
  elif r<=1.1*Rp and -0.9*DD-1.8*dd<z<0.9*DD+1.8*dd:
    max_area=2*(length/10.)**2
  else: 
    max_area=15*length**2*rho_sq/Rp**2
  return bool(area>max_area);

# Erzeuge das Netz
p,t,BouE,li_BE,bou_elem,CuE,li_CE=mt.DoTriMesh(p,v,edge_length=length,tri_refine=myrefine)



## Randkurven
Ps=[P9,P2,P5,P6,P9]
bseg=mt.RetrieveSegments(p,BouE,li_BE,Ps,['Segments','Nodes','Segments','Nodes'])
mt.PlotBoundary(p,bseg[0],'Segments')
mt.PlotBoundary(p,bseg[1],'Nodes')
mt.PlotBoundary(p,bseg[2],'Segments')
mt.PlotBoundary(p,bseg[3],'Nodes')
plt.show()


print("Nodes",len(p))
print("Elements",len(t))
 





