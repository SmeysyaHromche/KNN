

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    
    from src.learn.database.ocrdataset import OcrDataset
    
    dataset = OcrDataset(
        path_to_db="/home/xkukht01/Dev/KNN/.data/ocr_dataset/data.mdb",
        path_to_meta_db="/home/xkukht01/Dev/KNN/.data/ocr_dataset/meta.mdb"
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    for i in range(5):
        value, label = dataset[i]
        print(f"Sample {i}: Label={label}, Value Length={len(value)}")
    
    dataset.close_resources()