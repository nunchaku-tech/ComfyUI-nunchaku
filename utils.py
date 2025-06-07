from importlib.metadata import PackageNotFoundError, distribution, metadata


def get_package_metadata(package_name) -> str:
    try:
        meta = metadata(package_name)
        meta_dict = dict(meta)

        dist = distribution(package_name)
        location = dist.locate_file("").as_posix()  # 转为字符串路径

        lines = [f"{k}: {v}" for k, v in meta_dict.items()]
        lines.append(f"Location: {location}")
        return "\n".join(lines)

    except PackageNotFoundError:
        return f"Package '{package_name}' not found."


def get_package_version(package_name) -> str:
    try:
        meta = metadata(package_name)
        meta_dict = dict(meta)

        version = meta_dict.get("Version", "Unknown version")
        return version

    except PackageNotFoundError:
        return f"Package '{package_name}' not found."


def get_plugin_version() -> str:
    pass
