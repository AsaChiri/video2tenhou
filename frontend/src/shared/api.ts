export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Video2Tenhou": "1",
          },
          body: JSON.stringify(body),
        },
  );
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Could not read the response. Try again.");
  }
  if (!response.ok) throw new Error(data.error || "Request failed. Try again.");
  return data;
}

export function upload(
  file: File,
  onProgress: (percent: number) => void,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/upload");
    request.setRequestHeader("X-Video2Tenhou", "1");
    request.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable)
        onProgress((event.loaded / event.total) * 100);
    };
    request.onerror = () =>
      reject(
        new Error("Could not copy the recording. Keep the app open and retry."),
      );
    request.onload = () => {
      try {
        const data = JSON.parse(request.responseText);
        if (request.status >= 400) throw new Error(data.error);
        resolve(data.path);
      } catch (error) {
        reject(error);
      }
    };
    request.send(file);
  });
}
