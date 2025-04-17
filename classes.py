import os
import os

base_path = r"C:\my_project2\alcogole"

# Проверяем существование базового пути
if not os.path.exists(base_path):
    print(f"Директория {base_path} не существует!")
    exit()

# Проходим по всем основным папкам (Vermouth, Gin и т.д.)
for main_folder in os.listdir(base_path):
    main_folder_path = os.path.join(base_path, main_folder)
    
    if os.path.isdir(main_folder_path):
        # Путь к папке learning внутри основной папки
        learning_path = os.path.join(main_folder_path, "learning")
        
        if os.path.exists(learning_path) and os.path.isdir(learning_path):
            # Создаём текстовый файл с именем основной папки
            output_file = os.path.join(base_path, f"{main_folder}.txt")
            
            # Получаем список ПАПОК внутри learning (не файлов!)
            subfolders = []
            for item in os.listdir(learning_path):
                item_path = os.path.join(learning_path, item)
                if os.path.isdir(item_path):
                    subfolders.append(item)
            
            # Сортируем папки по имени
            subfolders.sort()
            
            # Записываем в файл
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Категория: {main_folder}\n\n")
                f.write("Список подпапок:\n")
                for folder in subfolders:
                    f.write(f"{folder}\n")
            
            print(f"Создан файл: {output_file}")

print("Обработка завершена!")