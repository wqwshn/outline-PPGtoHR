# Python 开源依赖清单

统计范围：Python 代码中实际使用到的第三方开源包，以及项目开发/运行所依赖的编辑器与 Python 运行环境。  
版本优先采用当前 `ppg-hr` 环境中已安装的版本；`numba` 在当前环境未安装，因此使用项目在 `pyproject.toml` / `environment.yml` 中声明的最低版本约束。  
标准库和项目内模块未计入。

| 序号 | 开源软件名 | 开源软件版本 | 开源软件链接网站 | 许可证名称 | 是否以及如何履行开源软件义务 |
| --- | --- | --- | --- | --- | --- |
| 1 | NumPy | 2.4.3 | https://numpy.org | BSD-3-Clause | 是。仅通过依赖方式使用，未修改上游源码；如对外发布安装包或镜像，需保留许可证和版权声明。 |
| 2 | SciPy | 1.17.1 | https://scipy.org/ | BSD-3-Clause | 是。仅通过依赖方式使用，未修改上游源码；如对外发布安装包或镜像，需保留许可证和版权声明。 |
| 3 | pandas | 3.0.2 | https://pandas.pydata.org | BSD-3-Clause | 是。仅通过依赖方式使用，未修改上游源码；如对外发布安装包或镜像，需保留许可证和版权声明。 |
| 4 | Matplotlib | 3.10.8 | https://matplotlib.org | Matplotlib License（基于 PSF License） | 是。仅通过依赖方式使用，未修改上游源码；如对外发布安装包或镜像，需保留版权与许可声明。 |
| 5 | scikit-learn | 1.8.0 | https://scikit-learn.org | BSD-3-Clause | 是。仅通过依赖方式使用，未修改上游源码；如对外发布安装包或镜像，需保留许可证和版权声明。 |
| 6 | Optuna | 4.8.0 | https://optuna.org/ | MIT License | 是。仅通过依赖方式使用，未修改上游源码；如对外发布安装包或镜像，需保留许可证与版权声明。 |
| 7 | numba | >=0.59（项目声明，当前环境未安装） | https://numba.pydata.org/ | BSD-2-Clause | 是。代码中为可选加速依赖，当前运行时会回退到纯 Python 实现；若后续启用并对外分发，需保留对应许可证文本与版权声明。 |
| 8 | PySide6 | 6.11.0 | https://doc.qt.io/qtforpython-6/ | LGPLv3/GPLv3（Qt for Python Community Edition） | 是。当前仅作为 GUI 依赖使用；若对外分发桌面程序，需遵守 LGPLv3/GPLv3 要求，或改用商业授权，并随包附带许可文本。 |
| 9 | pytest | 9.0.3 | https://docs.pytest.org/ | MIT License | 是。仅用于测试代码；如对外分发测试环境或镜像，需保留许可证与版权声明。 |
| 10 | Visual Studio Code | 1.117.0 | https://code.visualstudio.com/ | MIT License | 是。仅作为开发工具使用；若对外分发编辑器安装包或镜像，需保留许可证与版权声明。 |
| 11 | Python | 3.11.7 | https://www.python.org/ | Python Software Foundation License | 是。作为项目运行环境使用；若对外分发 Python 运行时或安装包，需保留 PSF License 与版权声明。 |

说明：

1. 以上“开源软件链接网站”采用各项目官网或官方文档首页。
2. “履行开源软件义务”按当前仓库源码使用方式填写，重点是“仅依赖调用、未修改上游源码”；若未来制作 wheel、安装包、Docker 镜像或桌面安装程序，应按对应许可证补充第三方声明文件。
