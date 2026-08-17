from __future__ import annotations

import os
import textwrap
from io import BytesIO
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from minio_service import MinioService


PLOT_TYPES = {"bar", "plot", "line", "pie", "table"}
CONTENT_TYPE = "image/png"


def _configure_chinese_font() -> None:
    preferred_fonts = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "SimSun",
)
    installed_fonts = {
        font_manager.FontProperties(fname=path).get_name()
        for path in font_manager.findSystemFonts()
    }
    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


def _get_minio_service(bucket_name: str) -> MinioService:
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("缺少 MINIO_ACCESS_KEY 或 MINIO_SECRET_KEY 环境变量")

    return MinioService(
        minio_endpoint=os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        minio_access_key=access_key,
        minio_secret_key=secret_key,
        minio_bucket_name=bucket_name,
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def _normalize_object_name(object_name: str) -> str:
    name = str(object_name).strip().replace("\\", "/").lstrip("/")
    if not name:
        raise ValueError("object_name 不能为空")
    if ".." in name.split("/"):
        raise ValueError("object_name 不能包含上级目录")
    if not name.lower().endswith(".png"):
        name += ".png"
    return name


def _validate_common(
    needed_type: str,
    x_data: Sequence[Any],
    values: Sequence[Any],
) -> tuple[str, list[Any], list[Any]]:
    chart_type = str(needed_type).strip().lower()
    if chart_type == "line":
        chart_type = "plot"
    if chart_type not in PLOT_TYPES:
        raise ValueError("不支持该图表类型，仅支持 bar、plot/line、pie 和 table")

    labels = list(x_data)
    raw_values = list(values)
    if not labels or not raw_values:
        raise ValueError("x_data 和 values 不能为空")
    if len(labels) != len(raw_values):
        raise ValueError("x_data 和 values 长度不一致")
    if len(labels) > 1000:
        raise ValueError("单次绘图的数据点不能超过 1000 个")

    if chart_type == "table":
        table_values = [
            str(value).strip()
            for value in raw_values
        ]
        if any(not value for value in table_values):
            raise ValueError("表格 values 不能包含空值")
        return chart_type, labels, table_values

    numeric_values: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool):
            raise ValueError(f"values[{index}] 不是有效数值")
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"values[{index}] 不是有效数值") from error

    if chart_type == "pie" and any(value < 0 for value in numeric_values):
        raise ValueError("饼图数据不能包含负数")
    if chart_type == "pie" and sum(numeric_values) <= 0:
        raise ValueError("饼图数据总和必须大于 0")

    return chart_type, labels, numeric_values


def _render_figure(
    chart_type: str,
    title: str,
    label: str,
    x_data: list[Any],
    values: list[Any],
) -> BytesIO:
    point_count = len(x_data)
    figure_width = max(10.0, min(24.0, point_count * 1.45))
    figure_height = 7.0 if chart_type in {"bar", "plot"} else 6.0

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    try:
        if chart_type == "bar":
            bars = ax.bar(
                x_data,
                values,
                width=0.6,
                label=label or None,
            )

            ax.set_xlabel("类别")
            ax.set_ylabel(label or "数值")

            value_labels = [
                f"{value:g}"
                for value in values
            ]

            ax.bar_label(
                bars,
                labels=value_labels,
                padding=3,
                fontsize=8,
                rotation=0,
            )

            ax.margins(y=0.12)

        elif chart_type == "plot":
            ax.plot(
                x_data,
                values,
                marker="o",
                label=label or None,
            )

            ax.set_xlabel("类别")
            ax.set_ylabel(label or "数值")
            ax.grid(True, alpha=0.3)

            for x, y in zip(x_data, values):
                ax.annotate(
                    f"{y:g}",
                    xy=(x, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

            ax.margins(y=0.12)

        elif chart_type == "pie":
            ax.pie(
                values,
                labels=[str(item) for item in x_data],
                autopct="%1.1f%%",
            )
            ax.axis("equal")

        else:
            ax.axis("off")

            table = ax.table(
                cellText=[
                    [str(x), str(value)]
                    for x, value in zip(x_data, values)
                ],
                colLabels=[
                    "类别",
                    label or "数值",
                ],
                cellLoc="center",
                loc="center",
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.4)

        if chart_type in {"bar", "plot"}:
            wrapped_labels = [
                "\n".join(
                    textwrap.wrap(
                        str(item),
                        width=20,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                )
                for item in x_data
            ]

            ax.set_xticks(range(point_count))
            ax.set_xticklabels(wrapped_labels)

            rotation = 35 if point_count >= 6 else 20

            plt.setp(
                ax.get_xticklabels(),
                rotation=rotation,
                ha="right",
                rotation_mode="anchor",
                fontsize=10,
            )

            ax.margins(x=0.03)

        ax.set_title(
            str(title).strip() or "数据图表"
        )

        if label and chart_type in {"bar", "plot"}:
            ax.legend()

        fig.tight_layout(pad=1.4)

        image = BytesIO()

        fig.savefig(
            image,
            format="png",
            dpi=150,
            bbox_inches="tight",
        )

        image.seek(0)
        return image

    finally:
        plt.close(fig)

def plot(
    object_name: str,
    needed_type: str,
    title: str,
    label: str,
    x_data: list[Any],
    values: list[Any],
    bucket_name: str = "plot",
) -> tuple[bool, dict[str, Any]]:
    """绘制图表并上传至 MinIO。"""
    try:
        chart_type, labels, prepared_values = _validate_common(
            needed_type,
            x_data,
            values,
        )
        normalized_object_name = _normalize_object_name(object_name)
        image = _render_figure(
            chart_type,
            title,
            label,
            labels,
            prepared_values,
        )
        service = _get_minio_service(bucket_name)
        uploaded = service.upload_file(
            object_name=normalized_object_name,
            data=image,
            content_type=CONTENT_TYPE,
            bucket_name=bucket_name,
            length=image.getbuffer().nbytes,
        )

        return True, {
            "status": "success",
            "skill": "plot",
            "answer": {
                "text": f"已生成{str(title).strip() or '数据'}图表，共包含{len(labels)}个数据点。"
            },
            "data": {
                "x": labels,
                "series": [
                    {
                        "name": str(label).strip() or "数值",
                        "values": prepared_values,
                    }
                ],
            },
            "parameters": {
                "chart_type": chart_type,
                "title": str(title).strip() or "数据图表",
                "series_count": 1,
                "point_count": len(labels),
            },
            "artifacts": [
                {
                    "type": "chart",
                    "bucket_name": bucket_name,
                    "object_key": normalized_object_name,
                    "etag": getattr(uploaded, "etag", None),
                }
            ],
            "error": None,
        }
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        return False, {
            "status": "failed",
            "skill": "plot",
            "answer": {"text": f"图表生成失败：{error}"},
            "data": {},
            "parameters": {},
            "artifacts": [],
            "error": {
                "code": "PLOT_GENERATION_FAILED",
                "message": str(error),
            },
        }
