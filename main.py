from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import uvicorn
import os

app = FastAPI()

# 💡 修正 CORS：將 allow_credentials 改為 False，避免 Chrome 嚴格阻擋
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

ai_session = None

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...), post_processing: float = Form(0.5)):
    global ai_session 
    
    try:
        if ai_session is None:
            print("首次去背：正在載入 AI 模型 (u2netp)...請稍候")
            ai_session = new_session("u2netp")
            
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data))

        # 💡 加強防爆：將圖片最大邊長縮小至 500 像素，確保絕對不爆記憶體
        max_size = 500 
        if input_image.width > max_size or input_image.height > max_size:
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"圖片已自動等比例縮小至: {input_image.size}")

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
