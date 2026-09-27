import torch

from unet import UNet


def main():

    # Select GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Create U-Net model
    model = UNet(
        in_channels=3,
        out_channels=1
    )

    # Move model to GPU/CPU
    model = model.to(device)

    # Create a dummy RGB input
    x = torch.randn(
        2,
        3,
        256,
        256
    ).to(device)

    # Disable gradient calculation during testing
    with torch.no_grad():

        output = model(x)

    # Print shapes
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    # Count parameters
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", total_params)


if __name__ == "__main__":
    main()