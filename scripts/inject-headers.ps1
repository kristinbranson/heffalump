$folder = (pwd).Path;
$archive = ("*script*","*build*","*pdollar*","*SFMT-src-1.5.1");

Get-ChildItem -Path $folder -r  | 
    Where-Object {$_.Name -match "CMakeLists.txt"} |

        #   Where-Object { $_.PsIsContainer -and $_.FullName -notmatch 'build' } |
        #   Where-Object { $_.PsIsContainer -and $_.FullName -notmatch '.git' } |
        #   Where-Object { $_.PsIsContainer -and $_.FullName -notmatch 'pdollar' } |
        #   Where-Object { $_.PsIsContainer -and $_.FullName -notmatch 'SFMT-src-1.5.1' } |
        #   Where-Object { $_.PsIsContainer -and $_.FullName -notmatch 'script' } |
          
