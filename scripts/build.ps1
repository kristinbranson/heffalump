function msbuild($ver) {
    $regKey = "HKLM:\software\Microsoft\MSBuild\ToolsVersions\$ver"
    $regProperty = "MSBuildToolsPath"
    join-path -path (Get-ItemProperty $regKey).$regProperty -childpath "msbuild.exe"
}

if (-Not (Test-Path build)){
    $installdir=join-path (pwd) "install"
    mkdir build
    cd build
    cmake -G "Visual Studio 12 2013 Win64" -DCMAKE_INSTALL_PREFIX=$installdir ..    
    cd ..
    $conf_ret_code=$LASTEXITCODE
    if($conf_ret_code -ne 0){
        throw "CMake failed with exit code $LASTEXITCODE."
    }
}

write-host "Building ..."
cd build
&(msbuild("12.0")) heffalump.sln /m /nologo /clp:"Verbosity=minimal;ShowTimestamp;"
$build_ret_code=$LASTEXITCODE
cd ..

if($build_ret_code -ne 0){
    throw "MSBUILD failed with exit code $LASTEXITCODE."
}
write-host "...Done"

