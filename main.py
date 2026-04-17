import sys

from rich.console import Console
from rich.table import Table
from rich.traceback import install

from src.application.service import ClassifierService
from src.infrastructure.image_adapter import PILImageLoader
from src.infrastructure.timm_adapter import TimmClassifier

# Advanced error reporting
install(show_locals=True)
console = Console()


def main(
    image_path: str = "image.png", model_name: str = "vit_base_patch16_384"
) -> None:
    """Main entry point for image classification."""

    try:
        # 1. Initialize Adapters (Infrastructure)
        console.print(f"[bold blue]Initializing model:[/bold blue] {model_name}...")
        classifier = TimmClassifier(model_name=model_name)

        # Determine correct input size from the model
        input_size = classifier.get_input_size()
        console.print(f"[bold green]Detected input size:[/bold green] {input_size}")

        loader = PILImageLoader(size=input_size)

        # 2. Initialize Service (Application)
        service = ClassifierService(classifier=classifier, loader=loader)

        # 3. Execute Core Logic
        console.print(f"[bold blue]Processing image:[/bold blue] {image_path}...")
        result = service.classify_image(image_path)

        # 4. Present Results
        table = Table(title=f"Classification Results for '{image_path}'")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model", result.model_name)
        table.add_row("Top Prediction", result.top_prediction.label)
        table.add_row("Confidence", f"{result.top_prediction.confidence:.4f}")

        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    # Get image path from args if provided
    img = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    main(image_path=img)
