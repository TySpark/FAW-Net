import pickle as pkl
from pathlib import Path


def main() -> None:
    pkl_file = Path(__file__).parent / "pkl" / "ANH0106A.pkl"
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
        # Keys - ['target', 'matrix', 'param']
        # target: dict[float, (rxy, pxy, ryx, pyx)] -> 形状 (4,) 的 float64 tensor
        # matrix: 功率谱矩阵 -> 形状 dict[float, (N, 7, 7)] 的 float64 tensor
        # params: 输入数据 -> 形状 dict[float, (N, features)] 的 float64 tensor
        print(list(data.keys()))
        frequencies = list(data["target"].keys())
        print(f"target - {len(data['target'])}")
        print(f"matrix - {data['matrix'][frequencies[0]].shape}")
        print(f"param - {data['param'][frequencies[0]].shape}")


if __name__ == "__main__":
    main()
