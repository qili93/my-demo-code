#!/bin/bash

wget https://github.com/microsoft/vscode-cpptools/releases/download/v1.14.4/cpptools-linux-$(uname -m).vsix
wget https://github.com/gitkraken/vscode-gitlens/releases/download/v12.2.2/gitlens-12.2.2.vsix
wget https://github.com/microsoft/vscode-python/releases/download/2021.1.502429796/ms-python-release.vsix
wget -O twxs.cmake-0.0.17.vsix https://marketplace.visualstudio.com/_apis/public/gallery/publishers/twxs/vsextensions/cmake/0.0.17/vspackage
