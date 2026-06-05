from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color

def generate_uniform_colored_pdf(color_rgb, filename="uniform_color_A4.pdf"):
    """Generate a uniform colored PDF in A4 format"""
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    r, g, b = color_rgb
    if max(color_rgb) > 1.0:
        r, g, b = r/255.0, g/255.0, b/255.0
    
    c.setFillColor(Color(r, g, b))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.save()
    
    return filename


if __name__ == "__main__":
    # Example usage
    color = (119,72,80)  # Red color in RGB
    generate_uniform_colored_pdf(color, "utils/red_A4.pdf")