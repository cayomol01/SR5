from gl import Renderer, color, V3, V2
from texture import Texture
from shaders import flat, gourad

width = 1024
height = 800

rend = Renderer(width, height)

rend.active_shader = gourad
rend.active_texture = Texture("bullTexture.bmp")

rend.glLoadModel("bull.obj",
                 translate = V3(0, 0, -10),
                 scale = V3(0.01,0.01,0.01),
                 rotate = V3(0,90,0))

rend.glFinish("output.bmp")