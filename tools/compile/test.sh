arch=$(uname -i)
if [[ $arch == x86_64* ]]; then
    echo "X64 Architecture"
elif [[ $arch == i*86 ]]; then
    echo "X32 Architecture"
elif  [[ $arch == arm* ]]; then
    echo "ARM Architecture"
fi

