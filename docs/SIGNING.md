# Code-Signing & Notarisierung

Die Installer werden im Build-Workflow (`.github/workflows/build.yml`) erzeugt. **Ohne**
Zertifikate sind sie *unsigniert* — die App läuft trotzdem, aber Nutzer müssen beim ersten
Start einmal „trotzdem öffnen" bestätigen. Mit den folgenden Secrets signiert & notarisiert
der Workflow **automatisch**, dann fällt der Warnhinweis weg.

> Aktueller Stand: macOS wird **ad-hoc** signiert (App startet nach Rechtsklick → „Öffnen").
> Sobald die Secrets gesetzt sind, schaltet der Workflow auf echte Developer-ID-Signierung um.

## macOS (Developer ID + Notarisierung)

Voraussetzung: **Apple Developer Program** (99 $/Jahr). Dann in den Repo-Settings unter
*Settings → Secrets and variables → Actions* anlegen:

| Secret | Inhalt |
|---|---|
| `APPLE_CERT_P12_BASE64` | Dein „Developer ID Application"-Zertifikat als `.p12`, base64-kodiert: `base64 -i cert.p12 \| pbcopy` |
| `APPLE_CERT_PASSWORD` | Passwort des `.p12` |
| `APPLE_DEVELOPER_ID` | z. B. `Developer ID Application: Dein Name (TEAMID)` |
| `APPLE_ID` | deine Apple-ID (E-Mail) |
| `APPLE_TEAM_ID` | 10-stellige Team-ID (App Store Connect → Membership) |
| `APPLE_APP_PASSWORD` | **App-spezifisches** Passwort (appleid.apple.com → Anmeldung & Sicherheit) |

**Zertifikat erzeugen:** Xcode → Settings → Accounts → Manage Certificates → „+" →
*Developer ID Application*. Danach in der Schlüsselbund-App als `.p12` exportieren.

Der Workflow importiert das Zertifikat in einen temporären Keychain, signiert die `.app` mit
Hardened Runtime + Timestamp, lädt sie per `notarytool` zur Notarisierung hoch und „stapelt"
das Ticket an die App. Ergebnis: startet ohne Gatekeeper-Warnung.

## Windows (Authenticode)

Voraussetzung: ein **Code-Signing-Zertifikat** (OV/EV, z. B. von Sectigo/DigiCert). Noch nicht
im Workflow verdrahtet — bei Bedarf folgenden Schritt vor „Paketieren" ergänzen und Secrets
`WINDOWS_CERT_PFX_BASE64` + `WINDOWS_CERT_PASSWORD` setzen:

```yaml
- name: Windows signieren
  if: runner.os == 'Windows' && env.WINDOWS_CERT_PFX_BASE64 != ''
  shell: pwsh
  env:
    PFX: ${{ secrets.WINDOWS_CERT_PFX_BASE64 }}
    PFX_PW: ${{ secrets.WINDOWS_CERT_PASSWORD }}
  run: |
    [IO.File]::WriteAllBytes("cert.pfx", [Convert]::FromBase64String($env:PFX))
    & "C:/Program Files (x86)/Windows Kits/10/bin/x64/signtool.exe" sign `
      /f cert.pfx /p $env:PFX_PW /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
      dist/ForgePix/ForgePix.exe
```

## Linux

Linux-Binaries werden i. d. R. nicht signiert. Für mehr Vertrauen später optional ein
**AppImage** mit GPG-Signatur oder Flatpak/Snap.

---

Bis Zertifikate vorliegen, ist der „nicht signiert"-Hinweis harmlos: einmal Rechtsklick →
„Öffnen" (macOS) bzw. „Weitere Informationen → Trotzdem ausführen" (Windows).
