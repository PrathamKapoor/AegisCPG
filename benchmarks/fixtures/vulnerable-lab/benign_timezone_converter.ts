// Benign TypeScript server: pure timezone offset conversion.
const server = new Server();

server.registerTool({
    name: "convert_timezone",
    description: "Converts between timezones",
}, (req, res) => {
    const fromTz = req.params.from_tz || "UTC";
    const toTz = req.params.to_tz || "UTC";
    res({ from_tz: fromTz, to_tz: toTz, offset_hours: 0 });
});
