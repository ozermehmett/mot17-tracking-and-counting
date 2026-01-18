import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_table():
    sequences = ['MOT17-09', 'MOT17-02', 'MOT17-04']
    data = []
    
    for seq in sequences:
        eval_path = f'outputs/{seq}/evaluation.json'
        results_path = f'outputs/{seq}/results.json'
        
        if not os.path.exists(eval_path) or not os.path.exists(results_path):
            continue
        
        with open(eval_path) as f:
            eval_data = json.load(f)
        with open(results_path) as f:
            results_data = json.load(f)
        
        det = eval_data['detection']
        track = eval_data['tracking']
        count = results_data['counts']
        
        data.append([
            seq,
            f"{det['precision']*100:.1f}%",
            f"{det['recall']*100:.1f}%",
            f"{det['f1']:.3f}",
            track['id_switches'],
            track['fragmentations'],
            count['entry'],
            count['exit'],
            count['total_crossings']
        ])
    
    # Tablo oluştur
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Sequence', 'Precision', 'Recall', 'F1', 'ID Switches', 
               'Fragments', 'Entry', 'Exit', 'Total']
    
    table = ax.table(cellText=data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.12] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Row colors
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.savefig('outputs/results_table.png', dpi=300, bbox_inches='tight')
    print("outputs/results_table.png created")


if __name__ == '__main__':
    generate_table()
