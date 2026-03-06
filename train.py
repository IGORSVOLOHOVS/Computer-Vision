from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data='data/data.yaml',
        epochs=50,                  # Снизим количество эпох
        imgsz=20,                  # КРИТИЧНО: Уменьшаем размер до 320. 640 не влезет в твою RAM.
        batch=2,                    # КРИТИЧНО: Минимальный батч, чтобы не уйти в своп.
        device='cpu',               # Оставляем CPU. Настройка XPU здесь не даст профита.
        workers=0,                  # КРИТИЧНО: Отключаем многопроцессорность загрузчика данных для экономии памяти.
        name='cash_counter_yolov8n_run1',
        patience=10,                
    )

    print("Обучение завершено. Результаты сохранены в папке:", results.save_dir)

if __name__ == '__main__':
    main()