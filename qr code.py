import qrcode

def print_qr_code_as_text(url):
    # QR 코드 생성
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    # QR 코드 이미지 텍스트로 변환
    qr_matrix = qr.get_matrix()
    for row in qr_matrix:
        line = ''.join('██' if cell else '  ' for cell in row)
        print(line)

# URL을 입력
url = "https://65e876ccc4a54f139f.gradio.live"
print_qr_code_as_text(url)
