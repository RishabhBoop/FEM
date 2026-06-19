# -*- coding: utf-8 -*-
"""
Klausur WS1920 
Methode der finiten Elemente in 2D
Netz

"""

import numpy as np
import meshtools as mt
import matplotlib.pyplot as plt

# Abmessungen
Ra=0.3
dd=0.01
Hm=0.08
Bm=0.08
ma=0.01
mb=0.05
rm0=0.095
zm0=0.
lls=0.001
sd=0.005
SH=0.01

length=0.01
epsL=length/100

p,t,BouE,li_BE,bou_elem,CuE,li_CE=mt.LoadTriMesh('KlausurWS1920_Netz.npz',show=True)

    
# Randkurven
Ps=[[0,Ra],[0,-Ra],[0,Ra]]
bseg=mt.RetrieveSegments(p,BouE,li_BE,Ps,['Nodes','Segments'])
mt.PlotBoundary(p,bseg[0],'Nodes')
mt.PlotBoundary(p,bseg[1],'Segments')
plt.show()

# Knotenidizes zu [R0,0], [R1,0], [R2,0]
Ps=[[dd,0],[Bm-dd,0],[Bm,0]]
fnodes=mt.FindClosestNode(range(len(p)),p,Ps)
node0=fnodes[0][0]
node1=fnodes[0][1]
node2=fnodes[0][2]

# r-Achse
Ps=[ [epsL,0],[Ra-epsL,0] ]
rAchse=mt.RetrieveSegments(p,CuE,li_CE,Ps,['Nodes'])
mt.PlotBoundary(p,rAchse[0],'Nodes')
plt.show()
rAchse=rAchse[0]

print("Nodes",len(p))
print("Elements",len(t))


    
 





