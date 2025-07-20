import os
import sys
from src.detector import ThirdYDetector


def main():
    data_folder = 'data'
    if not os.path.exists(data_folder):
        data_folder = '../data'
        if not os.path.exists(data_folder):
            print("Error: Data folder not found")
            sys.exit(1)

    output_dir = 'output' if data_folder == 'data' else '../output'
    os.makedirs(output_dir, exist_ok=True)

    try:
        detector = ThirdYDetector(data_folder)
        detector.load_and_sort()

        for start_id in ['1', '0', '2', '3']:
            result = detector.find_third_y(start_id, output_dir, debug=False)
            if result:
                print(f"Third lumbar vertebra found: {result}")
                return

        print("Third lumbar vertebra not detected")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()