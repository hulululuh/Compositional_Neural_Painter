from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def create_brush_stroke(texture_path, position, scale=1.0, rotation=0, color=(255, 0, 0), opacity=128):
    # 텍스처 로딩
    brush_texture = Image.open(texture_path).convert("RGBA")
    
    # 스케일 및 회전 적용
    brush_texture = brush_texture.resize((int(brush_texture.width * scale), int(brush_texture.height * scale)))
    brush_texture = brush_texture.rotate(rotation, expand=1)
    
    # 색상 및 투명도 적용
    brush_texture = np.array(brush_texture)
    red, green, blue, alpha = brush_texture.T
    brush_texture = np.array([color[0], color[1], color[2], alpha//2]).T
    brush_texture = Image.fromarray(brush_texture, 'RGBA')
    
    # 프레임 버퍼에 브러시 스트로크 적용
    frame_buffer = Image.new("RGBA", (400, 400), (255, 255, 255, 255))
    frame_buffer.paste(brush_texture, position, brush_texture)
    
    return frame_buffer
