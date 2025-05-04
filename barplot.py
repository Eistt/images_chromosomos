import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import cv2

def plot_bar_chart(class_counts, image_height):
    def chromosome_sort(name):
        try:
            numeric_part = ''.join(c for c in name if c.isdigit())
            if numeric_part:
                return int(numeric_part)
            elif name == 'X':
                return 1000
            elif name == 'Y':
                return 1001
        except:
            pass
        return 999 

    sorted_items = sorted(class_counts.items(), key=lambda x: chromosome_sort(x[0]))
    names, counts = zip(*sorted_items) if sorted_items else ([], [])


    fig, ax = plt.subplots(figsize=(4, 6))
    ax.barh(names, counts, color='green')
    ax.set_xlabel('Count')
    ax.set_title('Chromosome Count')

    for i, v in enumerate(counts):
        ax.text(v + 0.1, i, str(v), color='black', va='center')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    chart = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)

    chart_image = cv2.imdecode(chart, cv2.IMREAD_COLOR)
    chart_image = cv2.cvtColor(chart_image, cv2.COLOR_RGB2BGR)
    chart_image = cv2.resize(chart_image, (400, image_height))
    
    return chart_image
