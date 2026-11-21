Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
appPy = repoDir & "\mechi\app.py"

pyw = sh.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Python\Python312\pythonw.exe"
If Not fso.FileExists(pyw) Then
    pyw = "pythonw.exe"
End If

sh.CurrentDirectory = repoDir
sh.Run """" & pyw & """ """ & appPy & """", 0, False
