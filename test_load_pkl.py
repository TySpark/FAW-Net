import pickle as pkl
from pathlib import Path


def main() -> None:
    pkl_file = Path(__file__).parent / "pkl" / "ANH0106A.pkl"
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
        # Keys - ['target', 'matrix', 'param']
        # target: dict[float, (rxy, pxy, ryx, pyx)] -> float64 tensor of shape (4,)
        # matrix: power-spectrum matrices -> float64 tensor of shape dict[float, (N, 7, 7)]
        # params: input features -> float64 tensor of shape dict[float, (N, features)]
        print(list(data.keys()))
        frequencies = list(data["target"].keys())
        print(f"target - {len(data['target'])}")
        print(f"matrix - {data['matrix'][frequencies[0]].shape}")
        print(f"param - {data['param'][frequencies[0]].shape}")


if __name__ == "__main__":
    main()
