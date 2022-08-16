#Programa para crear una ventana de Graficos que se represente a través de un mapa de bits
#El programa fue creado con la ayuda del catedrático Carlos Alonso y es casi una copia de lo hecho en clase



import struct
from collections import namedtuple
import random
import numpy as np
from obj import Obj
from mathlib import Mathlib
from math import cos, sin, pi, tan

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
V4 = namedtuple('Point4', ['x', 'y', 'z', 'w'])

def char(c):
    return struct.pack('=c', c.encode('ascii'))

def word(w):
    return struct.pack('=h', w)

def dword(d):
    return struct.pack('=l', d)

def color(r,g,b):
    return bytes([int(b*255),int(g*255),int(r*255)])


def baryCoords(A, B, C, P):
    
    areaPBC = (B.y - C.y) * (P.x - C.x) + (C.x - B.x) * (P.y - C.y)
    areaPAC = (C.y - A.y) * (P.x - C.x) + (A.x - C.x) * (P.y - C.y)
    areaABC = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y)

    try:
        # PBC / ABC
        u = areaPBC / areaABC
        # PAC / ABC
        v = areaPAC / areaABC
        # 1 - u - v
        w = 1 - u - v
    except:
        return -1, -1, -1
    else:
        return u, v, w


class Renderer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.clearColor = color(0, 0, 0)
        self.currColor = color(1, 1, 1)
        self.libreria = Mathlib()
        
        self.active_shader = None
        self.active_texture = None
        
        self.dirLight = V3(0,0,-1)
        
        self.glClear()
        self.glViewPort(0, 0, self.width, self.height)
        
        self.glViewMatrix()
        self.glViewPort(0,0,self.width, self.height)
        
        
        
    def glClear(self):
        self.pixels = [[self.clearColor for y in range(self.height)] for x in range(self.width)]
        
        self.zbuffer = [[ float('inf') for y in range(self.height)]
                          for x in range(self.width)]
        
    def glViewMatrix(self, translate = V3(0,0,0), rotate = V3(0,0,0)):
        self.camMatrix = self.glCreateObjectMatrix(translate, rotate)
        N = len(self.camMatrix)
        inv = [None for _ in range(N)]
        
        for i in range(N):
            inv[i] = [None for _ in range(N)]

        self.viewMatrix = self.libreria.inverse(self.camMatrix, N, inv)
    
    def glLookAt(self, eye, camPosition = V3(0,0,0)):
        forward = self.libreria.subtract(camPosition, eye)
        forward = forward / np.norm(forward)

        right = self.libreria.cross(V3(0,1,0), forward)
        right = right / np.linalg.norm(right)

        up = self.libreria.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        

            

        self.camMatrix = [[right[0],up[0],forward[0],camPosition[0]],
                                    [right[1],up[1],forward[1],camPosition[1]],
                                    [right[2],up[2],forward[2],camPosition[2]],
                                    [0,0,0,1]]
        
        N = len(self.camMatrix)
        inv = [None for _ in range(N)]
        
        for i in range(N):
            inv[i] = [None for _ in range(N)]

        self.viewMatrix = self.libreria.inverse(self.camMatrix, N, inv)
        
    def glProjectionMatrix(self, n = 0.1, f = 1000, fov = 60):
        aspectRatio = self.vpwidth / self.vpheight
        t = tan( (fov * pi / 180) / 2) * n
        r = t * aspectRatio

        self.projectionMatrix = [[n/r,0,0,0],
                                           [0,n/t,0,0],
                                           [0,0,-(f+n)/(f-n),-(2*f*n)/(f-n)],
                                           [0,0,-1,0]]
        
    def glClearColor(self, r, g, b):
        self.clearColor = color(r, g, b)
        
    def glColor(self, r, g, b):
        self.currColor = color(r, g, b)
        
    def glPoint(self, x, y, clr = None):
        if (0 <= x < self.width) and (0 <= y < self.height):
            self.pixels[x][y] = clr or self.currColor
            
    def glViewPort(self, posX, posY, width, height):
        self.vpx = posX
        self.vpy = posY
        self.vpwidth = width
        self.vpheight = height
        
        
        self.viewportMatrix = [[width/2,0,0,posX+width/2],
                                [0,height/2,0,posY+height/2],
                                [0,0,0.5,0.5],
                                [0,0,0,1]]

        self.glProjectionMatrix()
        
    def glClearViewport(self, clr = None):
        for x in range(self.vpx, self.vpx + self.vpwidth):
            for y in range(self.vpy, self.vpy + self.vpheight):
                self.glPoint(x,y,clr)
                
    def glPoint_vp(self, ndcX, ndcY, clr = None):
        if ndcX < -1 or ndcX > 1 or ndcY < -1 or ndcY > 1:
            return

        x = (ndcX + 1) * (self.vpwidth / 2) + self.vpx
        y = (ndcY + 1) * (self.vpheight / 2) + self.vpy

        x = int(x)
        y = int(y)
        
        self.glPoint(x,y,clr)
        
    def glCreateRotationMatrix(self, pitch = 0, yaw = 0, roll = 0):
            
        pitch *= pi/180
        yaw   *= pi/180
        roll  *= pi/180

        pitchMat = [[1, 0, 0, 0],
                    [0, cos(pitch),-sin(pitch), 0],
                    [0, sin(pitch), cos(pitch), 0],
                    [0, 0, 0, 1]]

        yawMat =    [[cos(yaw), 0, sin(yaw), 0],
                    [0, 1, 0, 0],
                    [-sin(yaw), 0, cos(yaw), 0],
                    [0, 0, 0, 1]]

        rollMat =   [[cos(roll),-sin(roll), 0, 0],
                    [sin(roll), cos(roll), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]

        return self.libreria.multiply([pitchMat, yawMat, rollMat])

        
        
    def glCreateObjectMatrix(self, translate = V3(0,0,0), rotate = V3(0,0,0), scale = V3(1,1,1)):


        rotation = self.glCreateRotationMatrix(rotate.x, rotate.y, rotate.z)
        translation = [[1, 0, 0, translate.x],
                [0, 1, 0, translate.y],
                [0, 0, 1, translate.z],
                [0, 0, 0, 1]]
        
        
        scaleMat = [[scale.x, 0, 0, 0],
                [0, scale.y, 0, 0],
                [0, 0, scale.z, 0],
                [0, 0, 0, 1]]
    
        
       
        
        matrixFinal = self.libreria.multiply([translation,rotation, scaleMat])

        

        return matrixFinal


    def glTransform(self, vertex, matrix):
        v = V4(vertex[0], vertex[1], vertex[2], 1)
        v1 = self.libreria.HorVert(v)
        vt = self.libreria.VertHor(self.libreria.multiply([matrix, v1]))
        vt = vt[0]
        vf = V3(vt[0] / vt[3],
                vt[1] / vt[3],
                vt[2] / vt[3])

        return vf

    def glDirTransform(self, dirVector, rotMatrix):
        v = V4(dirVector[0], dirVector[1], dirVector[2], 0)
        v = self.libreria.HorVert(v)
        vt = self.libreria.VertHor(self.libreria.multiply([rotMatrix, v]))
        vt = vt[0]
        vf = V3(vt[0],
                vt[1],
                vt[2])

        return vf
    
    def glCamTransform(self, vertex):
        v = V4(vertex[0], vertex[1], vertex[2], 1)
        vt = np.array(self.viewportMatrix) @ np.array(self.projectionMatrix) @ np.array(self.viewMatrix) @ v
        vt = vt.tolist()[0]
        vf = V3(vt[0] / vt[3],
                vt[1] / vt[3],
                vt[2] / vt[3])
        
        return vf



    def glLoadModel(self, filename, translate = V3(0,0,0), rotate = V3(0,0,0), scale = V3(1,1,1)):
        model = Obj(filename)
        modelMatrix = self.glCreateObjectMatrix(translate, rotate, scale)
        rotationMatrix = self.glCreateRotationMatrix(rotate[0], rotate[1], rotate[2])

        for face in model.faces:
            vertCount = len(face)

            v0 = model.vertices[ face[0][0] - 1]
            v1 = model.vertices[ face[1][0] - 1]
            v2 = model.vertices[ face[2][0] - 1]

            v0 = self.glTransform(v0, modelMatrix)
            v1 = self.glTransform(v1, modelMatrix)
            v2 = self.glTransform(v2, modelMatrix)

            A = self.glCamTransform(v0)
            B = self.glCamTransform(v1)
            C = self.glCamTransform(v2)

            vt0 = model.texcoords[face[0][1] - 1]
            vt1 = model.texcoords[face[1][1] - 1]
            vt2 = model.texcoords[face[2][1] - 1]

            vn0 = model.normals[face[0][2] - 1]
            vn1 = model.normals[face[1][2] - 1]
            vn2 = model.normals[face[2][2] - 1]
            vn0 = self.glDirTransform(vn0, rotationMatrix)
            vn1 = self.glDirTransform(vn1, rotationMatrix)
            vn2 = self.glDirTransform(vn2, rotationMatrix)

            self.glTriangle_bc(A, B, C,
                               verts = (v0, v1, v2),
                               texCoords = (vt0, vt1, vt2),
                               normals = (vn0, vn1, vn2))

            if vertCount == 4:
                v3 = model.vertices[ face[3][0] - 1]
                v3 = self.glTransform(v3, modelMatrix)
                D = self.glCamTransform(v3)
                vt3 = model.texcoords[face[3][1] - 1]
                vn3 = model.normals[face[3][2] - 1]
                vn3 = self.glDirTransform(vn3, rotationMatrix)

                self.glTriangle_bc(A, C, D,
                                   verts = (v0, v2, v3),
                                   texCoords = (vt0, vt2, vt3),
                                   normals = (vn0, vn2, vn3))

    
    def glTriangle_std(self, A, B, C, clr = None):
            
        if A.y < B.y:
            A, B = B, A
        if A.y < C.y:
            A, C = C, A
        if B.y < C.y:
            B, C = C, B

        self.glLine(A,B, clr)
        self.glLine(B,C, clr)
        self.glLine(C,A, clr)

        
        def flatBottom(vA,vB,vC):
            try:
                mBA = (vB.x - vA.x) / (vB.y - vA.y)
                mCA = (vC.x - vA.x) / (vC.y - vA.y)
            except:
                pass
            else:
                x0 = vB.x
                x1 = vC.x
                for y in range(int(vB.y), int(vA.y)):
                    self.glLine(V2(x0, y), V2(x1, y), clr)
                    x0 += mBA
                    x1 += mCA

        def flatTop(vA,vB,vC):
            try:
                mCA = (vC.x - vA.x) / (vC.y - vA.y)
                mCB = (vC.x - vB.x) / (vC.y - vB.y)
            except:
                pass
            else:
                x0 = vA.x
                x1 = vB.x
                for y in range(int(vA.y), int(vC.y), -1):
                    self.glLine(V2(x0, y), V2(x1, y), clr)
                    x0 -= mCA
                    x1 -= mCB

        if B.y == C.y:
            # Parte plana abajo
            flatBottom(A,B,C)
        elif A.y == B.y:
            # Parte plana arriba
            flatTop(A,B,C)
        else:
            # Dibujo ambos tipos de triangulos
            # Teorema de intercepto
            D = V2( A.x + ((B.y - A.y) / (C.y - A.y)) * (C.x - A.x), B.y)
            flatBottom(A,B,D)
            flatTop(B,D,C)
    
    
    def glLine(self, v0, v1, clr = None):
            # Bresenham line algorithm
        # y = m * x + b
        x0 = int(v0.x)
        x1 = int(v1.x)
        y0 = int(v0.y)
        y1 = int(v1.y)

        # Si el punto0 es igual al punto 1, dibujar solamente un punto
        if x0 == x1 and y0 == y1:
            self.glPoint(x0,y0,clr)
            return

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        # Si la linea tiene pendiente mayor a 1 o menor a -1
        # intercambio las x por las y, y se dibuja la linea
        # de manera vertical
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        # Si el punto inicial X es mayor que el punto final X,
        # intercambio los puntos para siempre dibujar de 
        # izquierda a derecha       
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        offset = 0
        limit = 0.5
        m = dy / dx
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                # Dibujar de manera vertical
                self.glPoint(y, x, clr)
            else:
                # Dibujar de manera horizontal
                self.glPoint(x, y, clr)

            offset += m

            if offset >= limit:
                if y0 < y1:
                    y += 1
                else:
                    y -= 1
                
                limit += 1
                
    def glTriangle_bc(self, A, B, C, texCoords = (), normals = (), clr = None):
        # bounding box
        minX = round(min(A.x, B.x, C.x))
        minY = round(min(A.y, B.y, C.y))
        maxX = round(max(A.x, B.x, C.x))
        maxY = round(max(A.y, B.y, C.y))

        triangleNormal = self.libreria.cross( self.libreria.substract(B, A), self.libreria.substract(C,A))
        # normalizar
        triangleNormal = triangleNormal / np.linalg.norm(triangleNormal)


        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                u, v, w = baryCoords(A, B, C, V2(x, y))

                if 0<=u and 0<=v and 0<=w:

                    z = A.z * u + B.z * v + C.z * w

                    if 0<=x<self.width and 0<=y<self.height:
                        if z < self.zbuffer[x][y]:
                            self.zbuffer[x][y] = z

                            if self.active_shader:
                                r, g, b = self.active_shader(self,
                                                            baryCoords=(u,v,w),
                                                            vColor = clr or self.currColor,
                                                            texCoords = texCoords,
                                                            normals = normals,
                                                            triangleNormal = triangleNormal)



                                self.glPoint(x, y, color(r,g,b))
                            else:
                                self.glPoint(x,y, clr)
                                
    def glFinish(self, filename):
        with open(filename, "wb") as file:
        #header
            file.write(bytes('B'.encode('ascii')))
            file.write(bytes('M'.encode('ascii')))
            file.write(dword(14 + 40 + self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(14 + 40))
            
            
            #info header
            file.write(dword(40))
            file.write(dword(self.width))
            file.write(dword(self.height))
            file.write(word(1))
            file.write(word(24))
            file.write(dword(0))
            file.write(dword(self.width * self.height * 3))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            file.write(dword(0))
            
            
            #Color Tables
            for y in range(self.height):
                for x in range(self.width):
                    file.write(self.pixels[x][y])
                    
            

'''     #Ayuda obtenida de https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule#:~:text=If%20this%20number%20is%20odd,to%20fill%20in%20strange%20ways. Utililzando la even odd rule para saber si un punto es boundary o no
    def boundaries(self, x: int, y: int, poly) -> bool:
        num = len(poly)
        j = num - 1
        c = False
        for i in range(num):
            if (x == poly[i][0]) and (y == poly[i][1]):
                #El punto que se esta viendo es una esquina
                return True
            
            #El punto que se esta biendo está dentro de la figura tomando en cuenta la pendiente de  los puntos
            if ((poly[i][1] > y) != (poly[j][1] > y)):
                slope = (x-poly[i][0])*(poly[j][1]-poly[i][1])-(poly[j][0]-poly[i][0])*(y-poly[i][1])
                if slope == 0:
                    return True
                #REvisa si el slope es una linea horizontal y por lo tanto seria considerado un boundary
                if (slope < 0) != (poly[j][1] < poly[i][1]):
                    #Revisa si el punto que se está viendo está dentro de la figura
                    c = not c
            j = i
        return c
    
    def glFill(self,poly,color = None):
        for i in range(self.height):
            for j in range(self.width):
                if  self.boundaries(i,j,poly):
                    self.glPoint(i,j, clr = color)
                    
    def randomGlFill(self,poly):
        for i in range(self.height):
            for j in range(self.width):
                if self.boundaries(i,j,poly):
                    coolor = color(random.random(),random.random(),random.random())
                    self.glPoint(i,j, clr = coolor)
                 '''
        
        
        
    