use std::{
    io,
    path::PathBuf,
    process::{Child, Command, Stdio},
    str::FromStr,
};

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// A key that runs a shell command on the picked images, from `--bind KEY:COMMAND`.
#[derive(Clone, Debug)]
pub struct Bind {
    /// The key as you wrote it, like `ctrl-d`.
    pub name: String,
    modifiers: KeyModifiers,
    code: KeyCode,
    command: String,
}

impl FromStr for Bind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let (name, command) = s
            .split_once(':')
            .ok_or("expected KEY:COMMAND, for example 'ctrl-d:cp -- \"$@\" ~/picked/'")?;
        if command.trim().is_empty() {
            return Err(format!("{name} has no command"));
        }
        let (modifiers, code) = key(&name.to_ascii_lowercase()).ok_or(format!(
            "unknown key '{name}': use ctrl-a to ctrl-z, alt-a to alt-z, or f1 to f12. \
             Letters alone pick images, ctrl-c quits, and terminals send ctrl-h, ctrl-i, \
             ctrl-j and ctrl-m as other keys"
        ))?;
        Ok(Self {
            name: name.to_owned(),
            modifiers,
            code,
            command: command.to_owned(),
        })
    }
}

fn key(name: &str) -> Option<(KeyModifiers, KeyCode)> {
    let letter = |s: &str| match s.as_bytes() {
        [c @ b'a'..=b'z'] => Some(char::from(*c)),
        _ => None,
    };
    if let Some(rest) = name.strip_prefix("ctrl-") {
        let c = letter(rest).filter(|c| !"chijm".contains(*c))?;
        Some((KeyModifiers::CONTROL, KeyCode::Char(c)))
    } else if let Some(rest) = name.strip_prefix("alt-") {
        Some((KeyModifiers::ALT, KeyCode::Char(letter(rest)?)))
    } else {
        let n = name.strip_prefix('f')?.parse().ok()?;
        (1..=12)
            .contains(&n)
            .then_some((KeyModifiers::NONE, KeyCode::F(n)))
    }
}

impl Bind {
    pub fn matches(&self, key: KeyEvent) -> bool {
        let code = match key.code {
            KeyCode::Char(c) => KeyCode::Char(c.to_ascii_lowercase()),
            code => code,
        };
        code == self.code && key.modifiers.difference(KeyModifiers::SHIFT) == self.modifiers
    }

    /// Runs the command in the background, with the paths as its arguments: `"$1"`, or all of
    /// them as `"$@"`. A path never becomes part of the shell code, whatever its name. The
    /// picker owns the terminal, so the command gets no terminal.
    pub fn spawn(&self, paths: &[PathBuf]) -> io::Result<Child> {
        Command::new("sh")
            .args(["-c", &self.command, "scry"])
            .args(paths)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::testing::Temp;

    fn bind(s: &str) -> Result<Bind, String> {
        s.parse()
    }

    #[test]
    fn keys_avoid_the_picker_keys() {
        for good in ["ctrl-d:x", "CTRL-Y:x", "alt-o:x", "f1:x", "f12:a:b"] {
            assert!(bind(good).is_ok(), "{good}");
        }
        for bad in [
            "d:x", "D:x", "ctrl-c:x", "ctrl-m:x", "ctrl-1:x", "alt-:x", "f0:x", "f13:x", "ctrl-d",
            "ctrl-d: ",
        ] {
            assert!(bind(bad).is_err(), "{bad}");
        }
        assert_eq!(bind("f12:a:b").unwrap().command, "a:b");
    }

    #[test]
    fn matches_the_key_with_or_without_shift() {
        let ctrl_d = bind("ctrl-d:x").unwrap();
        let ctrl = KeyModifiers::CONTROL;
        assert!(ctrl_d.matches(KeyEvent::new(KeyCode::Char('d'), ctrl)));
        assert!(ctrl_d.matches(KeyEvent::new(
            KeyCode::Char('D'),
            ctrl | KeyModifiers::SHIFT
        )));
        assert!(!ctrl_d.matches(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE)));
        assert!(!ctrl_d.matches(KeyEvent::new(KeyCode::Char('d'), ctrl | KeyModifiers::ALT)));
        assert!(bind("f5:x").unwrap().matches(KeyEvent::from(KeyCode::F(5))));
    }

    #[test]
    fn paths_are_arguments_not_code() {
        let dir = Temp::dir("bind");
        let out = dir.join("out");
        let command = format!("ctrl-d:printf '%s|' \"$@\" > '{}'", out.display());
        let paths = [dir.join("$(echo pwned) 'q'.png"), dir.join("b.png")];
        let status = bind(&command)
            .unwrap()
            .spawn(&paths)
            .unwrap()
            .wait()
            .unwrap();
        assert!(status.success());
        let want = format!("{}|{}|", paths[0].display(), paths[1].display());
        assert_eq!(fs::read_to_string(out).unwrap(), want);
    }
}
