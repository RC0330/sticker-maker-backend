from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 換成輕量口袋版且穩定的模型 u2netp ---
print("正在初始化穩定版模型 (u2netp)...")
session = new_session("u2netp") 

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...), post_processing: float = Form(0.5)):
    try:
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data))
        
        # 為了防止記憶體爆炸，如果圖片太大，我們先稍微縮小它（選配）
        # if input_image.width > 2000:
        #     input_image.thumbnail((1500, 1500))

        print(f"正在執行穩定去背運算...")
        
        # --- 核心修改：關閉 alpha_matting 以節省記憶體 ---
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=False  # 👈 關閉這個就不會噴 1.86GB 的錯誤了
        )
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        print("🎉 運算完成！")
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"\n!!! 發生錯誤：{e}")
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
