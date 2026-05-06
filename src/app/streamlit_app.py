"""Initial Streamlit entrypoint placeholder."""

from __future__ import annotations

import streamlit as st


DISCLAIMER = (
    "This project is for educational and research purposes only. It does not provide "
    "financial advice, investment recommendations, or trading signals."
)


def main() -> None:
    """Render the initial placeholder app."""

    st.set_page_config(page_title="Stock Research Copilot", layout="wide")
    st.title("Stock Research Copilot")
    st.info("The dashboard implementation is planned after the data and analysis modules.")
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
