"""Gradio chat and catalog UI backed by the Fashion API."""

import logging
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import gradio as gr
import httpx
from PIL import Image

from agent.agents.chat.config import settings

MISSING_IMAGE = str(Path(__file__).with_name("missing_image.svg"))
logger = logging.getLogger(__name__)


def _render_outfit(outfit: dict | None) -> str:
    logger.info("function_start function=_render_outfit")
    if not outfit:
        return "### Outfit\nNo outfit yet."
    lines = [
        f"### {outfit['outfit_name']}",
        f"**Occasion:** {outfit['occasion']}  ",
        f"**Style:** {outfit['style']}",
    ]
    for item in outfit.get("items", []):
        lines.append(
            f"- **{item['category']} — {item['item_name']}**: {item['color']}, "
            f"{item['material']}, {item['fit']}. {item['styling_instruction']}"
        )
    if outfit.get("styling_tips"):
        lines.append("\n**Styling tips:** " + " · ".join(outfit["styling_tips"]))
    return "\n".join(lines)


async def _render_products(matches: dict, user_id: str) -> tuple[list[tuple[object, str]], str]:
    logger.info("function_start function=_render_products match_count=%d", len(matches))
    gallery: list[tuple[object, str]] = []
    details: list[str] = []
    async with httpx.AsyncClient(timeout=30) as client:
        for match in matches.values():
            for product in match.get("products", []):
                image: object = MISSING_IMAGE
                image_url = product.get("image_url")
                if image_url:
                    try:
                        UUID(str(product["product_id"]))
                        response = await client.get(
                            image_url,
                            headers={"X-User-ID": user_id},
                        )
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).copy()
                    except (ValueError, KeyError, httpx.HTTPError, OSError):
                        logger.warning("Processed wardrobe image unavailable")
                gallery.append((image, product["name"]))
                details.append(
                    f"- **{product['name']}** — "
                    f"{product.get('category', '')} · {product.get('color', '')} "
                    f"(ID: `{product['product_id']}`)"
                )
    body = "\n".join(details) if details else "No matching uploaded wardrobe items."
    return gallery, "### Wardrobe matches\n" + body


async def send_message(message, history, user_id, thread_id, awaiting_clarification):
    logger.info("function_start function=send_message")
    if not message.strip():
        return (
            history, "", gr.skip(), gr.skip(), gr.skip(),
            thread_id, awaiting_clarification, user_id,
        )
    user_id = user_id.strip() or f"ui-{uuid4()}"
    thread_id = thread_id or str(uuid4())
    headers = {"X-User-ID": user_id}
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{settings.api_base_url}/v1/chat",
                headers=headers,
                json={
                    "thread_id": thread_id,
                    "user_input": message,
                    "resume": bool(awaiting_clarification),
                },
            )
            response.raise_for_status()
            result = response.json()
        answer = result.get("clarification") if result.get("interrupted") else result.get("message")
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer or "No response was returned."},
        ]
        gallery, product_text = await _render_products(
            result.get("product_matches", {}), user_id
        )
        return (
            history, "", _render_outfit(result.get("outfit")), product_text,
            gallery, thread_id, bool(result.get("interrupted")), user_id,
        )
    except Exception as exc:  # noqa: BLE001 - keep the chat UI responsive to service failures
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Service error: {exc}"},
        ]
        return (
            history, "", gr.skip(), gr.skip(), gr.skip(),
            thread_id, awaiting_clarification, user_id,
        )


with gr.Blocks(title="Fashion Assistant") as demo:
    gr.Markdown("# Fashion Assistant")
    thread_state = gr.State("")
    clarification_state = gr.State(False)
    with gr.Row():
        with gr.Column(scale=2):
            user_id_box = gr.Textbox(label="User ID", value=f"ui-{uuid4()}")
            chatbot = gr.Chatbot(type="messages", height=500)
            message_box = gr.Textbox(label="Message or feedback", lines=2)
            send = gr.Button("Send", variant="primary")
        with gr.Column(scale=2):
            outfit_view = gr.Markdown("### Outfit\nNo outfit yet.")
            product_view = gr.Markdown("### Products\nNo products yet.")
            gallery = gr.Gallery(label="Catalog products", columns=3, height=420)
    outputs = [
        chatbot, message_box, outfit_view, product_view, gallery,
        thread_state, clarification_state, user_id_box,
    ]
    inputs = [message_box, chatbot, user_id_box, thread_state, clarification_state]
    send.click(send_message, inputs, outputs)
    message_box.submit(send_message, inputs, outputs)


def launch() -> None:
    logger.info("function_start function=launch")
    auth = None
    if settings.ui_auth_user and settings.ui_auth_password:
        auth = (settings.ui_auth_user, settings.ui_auth_password)
    if settings.enable_ngrok:
        if auth is None:
            raise RuntimeError(
                "UI_AUTH_USER and UI_AUTH_PASSWORD are required for public ngrok access"
            )
        if not settings.ngrok_auth_token:
            raise RuntimeError("NGROK_AUTHTOKEN is required when ENABLE_NGROK=true")
        from pyngrok import ngrok
        ngrok.set_auth_token(settings.ngrok_auth_token)
        public_url = ngrok.connect(settings.ui_port).public_url
        print(f"ngrok URL: {public_url}")
    demo.launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        auth=auth,
    )


if __name__ == "__main__":
    launch()
