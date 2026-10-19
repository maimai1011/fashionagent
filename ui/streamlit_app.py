"""Streamlit profile, wardrobe, and chat UI for the unified API."""

import mimetypes
from uuid import UUID, uuid4

import httpx
import streamlit as st

try:
    from ui.config import default_api_url
except ModuleNotFoundError as exc:
    if exc.name != "ui":
        raise
    # Streamlit adds this script's directory, not the repository root, to sys.path.
    from config import default_api_url


def _json_object(response: httpx.Response) -> dict[str, object]:
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError("The API returned an invalid response")
    return result


def create_profile(
    api_url: str,
    user_id: str,
    email: str,
    age: int,
    gender: str,
) -> dict[str, object]:
    """Create a profile using the API's form-data contract."""
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{api_url.rstrip('/')}/v1/user-profile",
            data={
                "user_id": user_id,
                "email": email,
                "age": str(age),
                "gender": gender,
            },
        )
    return _json_object(response)


def upload_wardrobe_image(
    api_url: str,
    user_id: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> dict[str, object]:
    """Upload one clothing image and receive its processing item ID."""
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{api_url.rstrip('/')}/v1/wardrobe",
            headers={"X-User-ID": user_id},
            files={"image": (filename, content, content_type)},
        )
    return _json_object(response)


def get_wardrobe_result(api_url: str, user_id: str, item_id: str) -> dict[str, object]:
    """Read processing status for a wardrobe item owned by this user."""
    valid_id = str(UUID(item_id))
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{api_url.rstrip('/')}/v1/wardrobe/{valid_id}",
            headers={"X-User-ID": user_id},
        )
    return _json_object(response)


def get_processed_image(api_url: str, user_id: str, item_id: str) -> bytes:
    """Fetch the processed image through the authenticated API route."""
    valid_id = str(UUID(item_id))
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{api_url.rstrip('/')}/v1/wardrobe/{valid_id}/image",
            headers={"X-User-ID": user_id},
        )
    response.raise_for_status()
    return response.content


def send_chat(
    api_url: str,
    user_id: str,
    thread_id: str,
    user_input: str,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Send one turn using the API's current request contract."""
    with httpx.Client(timeout=210) as client:
        response = client.post(
            f"{api_url.rstrip('/')}/v1/chat",
            headers={"X-User-ID": user_id},
            json={
                "thread_id": thread_id,
                "user_input": user_input,
                "resume": resume,
            },
        )
    return _json_object(response)


def _error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        return f"API error {exc.response.status_code}: {detail or exc.response.text}"
    return str(exc)


def _new_conversation() -> None:
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.awaiting_clarification = False


def _show_outfit(outfit: dict[str, object]) -> None:
    st.markdown(f"**{outfit.get('outfit_name') or 'Outfit'}**")
    details = [str(outfit[key]) for key in ("occasion", "style") if outfit.get(key)]
    if details:
        st.caption(" · ".join(details))
    for item in outfit.get("items", []):
        name = item.get("item_name") or item.get("category") or "Item"
        attributes = [
            str(item[key]) for key in ("color", "material", "fit") if item.get(key)
        ]
        st.markdown(f"- **{name}** — {', '.join(attributes)}")
    tips = outfit.get("styling_tips") or []
    if tips:
        st.caption("Styling tips: " + " · ".join(str(tip) for tip in tips))


def _show_products(matches: dict[str, object], api_url: str, user_id: str) -> None:
    for match in matches.values():
        item = match.get("requested_item") or {}
        st.markdown(f"**{item.get('item_name') or 'Outfit item'}**")
        products = match.get("products") or []
        if not products:
            st.caption("No matching uploaded wardrobe items.")
        for product in products:
            name = product.get("category") or item.get("category") or "Wardrobe item"
            details = [
                str(product[key])
                for key in ("color", "material", "fit")
                if product.get(key)
            ]
            label = f"- **{name}**"
            if details:
                label += f" — {' · '.join(details)}"
            st.markdown(label)
            if product.get("image_url"):
                try:
                    st.image(
                        get_processed_image(api_url, user_id, str(product["product_id"])),
                        width=180,
                    )
                except (httpx.HTTPError, ValueError):
                    st.caption("Processed image unavailable.")


def _show_message(entry: dict[str, object], api_url: str, user_id: str) -> None:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        result = entry.get("result") or {}
        if result.get("outfit"):
            with st.expander("Outfit", expanded=False):
                _show_outfit(result["outfit"])
        if result.get("product_matches"):
            with st.expander("Matching wardrobe items", expanded=False):
                _show_products(result["product_matches"], api_url, user_id)


def _show_profile_page(api_url: str, user_id: str) -> None:
    st.header("Profile")
    st.caption("Create your profile before uploading wardrobe images. Profiles can be created once.")
    if not user_id:
        st.info("Enter a stable User ID in the sidebar first.")

    with st.form("profile_form"):
        email = st.text_input("Email address")
        age = st.number_input("Age", min_value=0, max_value=130, value=25, step=1)
        gender = st.selectbox("Gender", ["female", "male"], index=None)
        submitted = st.form_submit_button("Create profile")

    if submitted:
        if not user_id:
            st.error("User ID is required.")
        elif not email.strip() or gender is None:
            st.error("Email and gender are required.")
        else:
            try:
                profile = create_profile(api_url, user_id, email.strip(), int(age), gender)
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                st.error(_error_message(exc))
            else:
                st.session_state.profile = profile
                st.success("Profile created. You can now upload clothing in Wardrobe.")

    if profile := st.session_state.get("profile"):
        st.write(f"User ID: `{profile['user_id']}`")
        st.write(f"Email: {profile['email']}")
        st.write(f"Age: {profile['age']} · Gender: {profile['gender']}")


def _refresh_wardrobe_item(api_url: str, user_id: str, item_id: str) -> None:
    result = get_wardrobe_result(api_url, user_id, item_id)
    item = st.session_state.wardrobe_items.get(item_id, {}).copy()
    item.update(result)
    item.pop("status_error", None)
    if (
        result.get("status") == "completed"
        and result.get("processed_image_url")
        and not item.get("processed_image")
    ):
        try:
            item["processed_image"] = get_processed_image(api_url, user_id, item_id)
            item.pop("image_error", None)
        except (httpx.HTTPError, ValueError) as exc:
            item["image_error"] = _error_message(exc)
    st.session_state.wardrobe_items[item_id] = item


_PENDING_WARDROBE_STATUSES = frozenset({"uploaded", "queued", "processing"})


@st.fragment(run_every="5s")
def _show_wardrobe_statuses(api_url: str, user_id: str) -> None:
    """Poll pending items and redraw only the status/results area."""
    items = st.session_state.wardrobe_items
    refresh_all = st.button("Refresh now", disabled=not user_id or not items)
    for item_id, item in list(items.items()):
        if not user_id or not (refresh_all or item.get("status") in _PENDING_WARDROBE_STATUSES):
            continue
        try:
            _refresh_wardrobe_item(api_url, user_id, item_id)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            item["status_error"] = _error_message(exc)

    pending_count = sum(
        item.get("status") in _PENDING_WARDROBE_STATUSES for item in items.values()
    )
    if pending_count:
        st.caption(f"{pending_count} image(s) still pending. Status updates every 5 seconds.")

    for item_id, item in reversed(list(items.items())):
        status = item.get("status", "unknown")
        with st.container(border=True):
            st.subheader(item.get("original_filename") or "Clothing image")
            st.caption(f"Item ID: {item_id}")
            if status == "completed":
                st.success("Processing completed")
                st.write(f"Type: {item.get('clothing_type') or 'unknown'}")
                st.write(f"Dominant color: {item.get('dominant_color') or 'unknown'}")
            elif status == "processing":
                st.info("Processing image…")
            elif status in {"uploaded", "queued"}:
                st.info("Queued for processing…")
            elif status in {"failed", "queue_failed"}:
                st.error("Image processing failed")
            else:
                st.write(f"Status: {status}")
            if item.get("processed_image"):
                st.image(item["processed_image"], caption="Processed clothing image")
            if item.get("status_error"):
                st.warning(f"Could not refresh status: {item['status_error']}")
            if item.get("image_error"):
                st.warning(f"Could not load processed image: {item['image_error']}")
            if item.get("error_message"):
                st.error(item["error_message"])


def _show_wardrobe_page(api_url: str, user_id: str) -> None:
    st.header("Wardrobe")
    st.caption("Upload clothing images and watch their processing status below.")
    if not user_id:
        st.info("Enter the same User ID used for your profile in the sidebar.")

    with st.form("wardrobe_upload_form"):
        images = st.file_uploader(
            "Clothing images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Upload images")

    if submitted:
        if not user_id:
            st.error("User ID is required.")
        elif not images:
            st.error("Choose at least one image.")
        else:
            for image in images:
                content_type = (
                    image.type if image.type and image.type.startswith("image/")
                    else mimetypes.guess_type(image.name)[0] or "application/octet-stream"
                )
                try:
                    uploaded = upload_wardrobe_image(
                        api_url, user_id, image.name, image.getvalue(), content_type
                    )
                except (httpx.HTTPError, TypeError, ValueError) as exc:
                    st.error(f"{image.name}: {_error_message(exc)}")
                else:
                    st.session_state.wardrobe_items[uploaded["item_id"]] = uploaded
                    st.success(f"{image.name} queued. Item ID: {uploaded['item_id']}")

    with st.expander("Find an existing upload by item ID"):
        with st.form("find_wardrobe_item_form"):
            existing_id = st.text_input("Item ID")
            find_submitted = st.form_submit_button("Find item")
        if find_submitted:
            if not user_id:
                st.error("User ID is required.")
            else:
                try:
                    _refresh_wardrobe_item(api_url, user_id, existing_id.strip())
                except (httpx.HTTPError, TypeError, ValueError) as exc:
                    st.error(_error_message(exc))

    _show_wardrobe_statuses(api_url, user_id)


def _show_chat_page(api_url: str, user_id: str, thread_id: str) -> None:
    st.header("Chat")
    st.caption("Ask the local fashion agent for styling ideas.")
    context = (api_url, user_id, thread_id)
    if st.session_state.get("active_context") != context:
        st.session_state.messages = []
        st.session_state.awaiting_clarification = False
        st.session_state.active_context = context

    for entry in st.session_state.messages:
        _show_message(entry, api_url, user_id)

    if st.session_state.awaiting_clarification:
        st.info("Please answer the clarification question to continue this conversation.")

    prompt = st.chat_input(
        "Answer the question..." if st.session_state.awaiting_clarification else "Ask for an outfit...",
        max_chars=10_000,
        disabled=not user_id or not thread_id,
    )
    if not prompt:
        return

    user_entry = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_entry)
    _show_message(user_entry, api_url, user_id)
    with st.spinner("Thinking..."):
        try:
            result = send_chat(
                api_url,
                user_id,
                thread_id,
                prompt,
                resume=st.session_state.awaiting_clarification,
            )
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            error_entry = {
                "role": "assistant",
                "content": f"Chat request failed: {_error_message(exc)}",
            }
            st.session_state.messages.append(error_entry)
            _show_message(error_entry, api_url, user_id)
            return

    st.session_state.awaiting_clarification = bool(result.get("interrupted"))
    answer = (
        result.get("clarification")
        if st.session_state.awaiting_clarification
        else result.get("message")
    ) or "The assistant returned no text."
    assistant_entry = {"role": "assistant", "content": answer, "result": result}
    st.session_state.messages.append(assistant_entry)
    _show_message(assistant_entry, api_url, user_id)


def main() -> None:
    st.set_page_config(page_title="Fashion Assistant", page_icon="👗", layout="wide")
    st.title("Fashion Assistant")

    if "api_url" not in st.session_state:
        st.session_state.api_url = default_api_url()
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False
    if "wardrobe_items" not in st.session_state:
        st.session_state.wardrobe_items = {}

    with st.sidebar:
        st.header("Your account")
        api_url = st.text_input("API URL", key="api_url").strip().rstrip("/")
        user_id = st.text_input("User ID", key="user_id").strip()
        page = st.radio("Go to", ["Profile", "Wardrobe", "Chat"])
        st.divider()
        thread_id = st.text_input("Conversation ID", key="thread_id").strip()
        st.button("New conversation", on_click=_new_conversation)

    account_context = (api_url, user_id)
    if st.session_state.get("active_account_context") != account_context:
        st.session_state.profile = None
        st.session_state.wardrobe_items = {}
        st.session_state.messages = []
        st.session_state.awaiting_clarification = False
        st.session_state.active_account_context = account_context

    if not api_url:
        st.error("Set the API URL in the sidebar.")
        return
    if page == "Profile":
        _show_profile_page(api_url, user_id)
    elif page == "Wardrobe":
        _show_wardrobe_page(api_url, user_id)
    else:
        _show_chat_page(api_url, user_id, thread_id)


if __name__ == "__main__":
    main()
