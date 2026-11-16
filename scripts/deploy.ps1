
      - name: Prepare application deployment
        shell: powershell
        run: |
          powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\deploy.ps1"
