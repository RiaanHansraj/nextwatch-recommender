import argparse

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nextwatch",
        description="Recommend next shows from Netflix/Prime based on viewing history."
    )
    parser.add_argument("--history", type=str, help="Path to Netflix viewing history CSV")
    args = parser.parse_args()

    if not args.history:
        print("✅ OK: project runs. Next stage will read Netflix CSV.")
        return

    print(f"History path received: {args.history}")

if __name__ == "__main__":
    main()
