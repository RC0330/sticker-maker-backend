from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

ai_session = None

# 🌟 終極關鍵：移除了 async，讓超耗腦力的去背工作在「背景替身」執行，絕不卡死伺服器
@app.post("/remove-bg/")
def remove_bg(file: UploadFile = File(...), post_processing: float = Form(0.5)):
    global ai_session 
    
    try:
        if ai_session is None:
            print("首次去背：正在載入 AI 模型 (u2netp)...請稍候")
            ai_session = new_session("u2netp")
            
        # 配合移除 async，讀取檔案的方式改為同步讀取
        image_data = file.file.read() 
        input_image = Image.open(io.BytesIO(image_data))

        # 終極防爆：最大邊長 400 像素，保證 Render 免費主機絕對吃得消
        max_size = 400 
        if input_image.width > max_size or input_image.height > max_size:
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"圖片已自動縮小至: {input_image.size} 以防止主機當機")

        print("正在執行去背運算...")
        
        output_image = remove(
            input_image,
            session=ai_session,
            alpha_matting=False
        )
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        print("🎉 運算完成！")
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"\n!!! 發生錯誤：{e}")
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
