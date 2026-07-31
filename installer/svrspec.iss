; Inno Setup script for svrspec.
;
; Wraps the cx_Freeze folder build into a conventional setup.exe. Installs per
; user into LocalAppData so it never raises a UAC prompt -- this is an analysis
; tool, not something that needs the machine.
;
;   ISCC.exe installer\svrspec.iss

#define AppName "svrspec"
#define AppVersion "0.2.2"
#define AppExe "svrspec.exe"
#define BuildDir "..\build\exe.win-amd64-3.12"

[Setup]
AppId={{8C4E1F2A-7B93-4D6E-A5C8-2F91D0B7E534}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=rokaproj
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
OutputDir=..\dist
OutputBaseFilename=svrspec-{#AppVersion}-setup
SetupIconFile=svrspec.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
AppComments=GPU 서빙 서버 스펙 산정 시뮬레이터

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{#AppName} 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
