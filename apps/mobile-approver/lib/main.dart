/// AegisCPG mobile approver -- the FIDO2 approval screen.
///
/// Author: Pratham Kapoor <prathamkapoor027@gmail.com>
/// https://github.com/PrathamKapoor
///
/// WHAT
/// Renders a pending declassification: the source and target with their lattice
/// levels, the datum hash, a live countdown, and a passkey button that performs
/// the WebAuthn ceremony and reports the gateway's verdict.
///
/// WHY THE APPROVAL RUNS ON A SEPARATE DEVICE
/// The workstation is the thing being protected. A malicious MCP server that runs
/// on it can ask the local firewall for anything it likes; if the approval
/// ceremony were on the same machine, the server could simply render the dialog
/// itself and approve its own release. That is TV13, the consent-bypass. Tying
/// the assertion to a passkey on a device the server cannot reach makes the
/// approval an independent human act rather than a self-granted permission.
///
/// WHY THE PRIVATE KEY NEVER ENTERS DART
/// `flutter_web_auth_2` hands the challenge to the platform authenticator and
/// returns an assertion. There is no API here that could export the key, which
/// is the property that makes the approval worth having.
///
/// HOW
/// The decision logic is `policy.dart`, which has no widget imports and is the
/// same state machine `workflows/workflows.py` implements. This file renders it
/// and performs the platform call.
///
/// NOT BUILT ON THIS HOST
/// No Dart or Flutter SDK is installed on this air-gapped ARM64 machine, so
/// this package is not compiled or tested here. `policy.dart` is pure Dart with
/// no Flutter import and is the part that would be worth testing first.
library;

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_web_auth_2/flutter_web_auth_2.dart';

import 'policy.dart';

void main() {
  runApp(const AegisMobileApproverApp());
}

/// The relying-party identifier the platform authenticator will check.
///
/// Must match the origin the gateway serves its challenge from. A mismatch
/// produces a verification failure with no useful diagnostic, so it is a named
/// constant rather than a string literal in a call.
const String relyingPartyId = 'aegis.local';

/// The root widget.
class AegisMobileApproverApp extends StatelessWidget {
  const AegisMobileApproverApp({super.key, this.request});

  /// Injected so the widget tree can be built in a test without a channel.
  final DeclassificationRequest? request;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Aegis Mobile Approver',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF10B981),
          secondary: Color(0xFFFB7185),
          surface: Color(0xFF18181B),
        ),
        fontFamily: 'monospace',
      ),
      home: request == null
          ? const _WaitingScreen()
          : ApprovalScreen(request: request!),
    );
  }
}

/// Shown when no request has been handed to the app.
class _WaitingScreen extends StatelessWidget {
  const _WaitingScreen();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: const <Widget>[
              Icon(Icons.shield_outlined, size: 64, color: Color(0xFF10B981)),
              SizedBox(height: 16),
              Text(
                'No pending declassification',
                style: TextStyle(fontSize: 18, color: Color(0xFFA1A1AA)),
              ),
              SizedBox(height: 8),
              Text(
                'A request arrives from the gateway when a tool asks to move '
                'data across a trust boundary.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Color(0xFF71717A)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// One pending declassification, awaiting a passkey assertion.
class ApprovalScreen extends StatefulWidget {
  const ApprovalScreen({super.key, required this.request});

  final DeclassificationRequest request;

  @override
  State<ApprovalScreen> createState() => _ApprovalScreenState();
}

class _ApprovalScreenState extends State<ApprovalScreen> {
  late final ApprovalModal _modal = ApprovalModal(
    widget.request,
    challengeId: 'cf_${widget.request.sessionId}',
  );

  Timer? _ticker;
  DateTime _now = DateTime.now();
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    // A one-second tick is enough for a countdown whose deadline is minutes
    // away, and it keeps the UI thread idle the rest of the time.
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() => _now = DateTime.now());
      }
    });
  }

  @override
  void dispose() {
    _ticker?.cancel();
    super.dispose();
  }

  /// Performs the WebAuthn ceremony and hands the gateway's verdict back.
  ///
  /// The device does not verify its own signature: it has never met the relying
  /// party. It performs the challenge, then the gateway -- which holds the
  /// registered public key -- says whether the assertion was good, and the
  /// modal resolves from that.
  Future<void> _approve() async {
    if (!_modal.canApprove(_now) || _busy) {
      return;
    }
    setState(() => _busy = true);
    try {
      final response = await FlutterWebAuth2.authenticate(
        url: 'https://$relyingPartyId/aegis/step-up',
        callbackUrlScheme: 'aegis',
        data: widget.request.bindingDigest(_modal.challengeId),
        preferEphemeral: true,
      );
      final parameters = Uri.parse(response.url).queryParameters;
      final verdict = _modal.resolve(
        now: DateTime.now(),
        presentedChallengeId: parameters['challenge_id'] ?? '',
        signatureValid: parameters['verified'] == 'true',
        bindingMatches: parameters['binding'] == 'match',
        clientId: parameters['client_id'] ?? '',
      );
      if (mounted) {
        setState(() {});
        _showVerdict(verdict);
      }
    } catch (error) {
      if (mounted) {
        setState(() {});
        _showVerdict(
          _modal.resolve(
            now: DateTime.now(),
            presentedChallengeId: '',
            signatureValid: false,
            bindingMatches: false,
            clientId: '',
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  void _showVerdict(ApprovalVerdict verdict) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: verdict.isAllowed
            ? const Color(0xFF065F46)
            : const Color(0xFF881337),
        content: Text(
          verdict.isAllowed
              ? 'Released. ${verdict.bindingDigest.substring(0, 16)}…'
              : 'Denied: ${verdict.reason?.wire}',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final request = widget.request;
    final remaining = _modal.secondsRemaining(_now);
    final canApprove = _modal.canApprove(_now);
    final expired = remaining <= 0 && !_modal.isTerminal;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Step-up approval'),
        backgroundColor: const Color(0xFF18181B),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _row('source', '${request.sourceServer}  ${request.sourceLevel.label}'),
            _row('target', '${request.targetServer}  ${request.targetLevel.label}'),
            _row(
              'datum',
              'sha256:${request.datumHash.substring(0, 24)}…',
            ),
            _row('expires in', expired ? 'expired' : '${remaining}s'),
            if (request.isDowngrade)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: Text(
                  'This release LOWERS the classification of data you are not '
                  'cleared to read.',
                  style: TextStyle(color: Color(0xFFFB7185), fontSize: 12),
                ),
              ),
            const Spacer(),
            if (_modal.isTerminal)
              _verdictPanel(_modal.verdict!)
            else
              CheckboxListTile(
                value: _modal.isAcknowledged,
                onChanged: expired ? null : (_) => setState(_modal.acknowledge),
                controlAffinity: ListTileControlAffinity.leading,
                title: const Text(
                  'I have read the source, target and datum hash, and authorise '
                  'this release.',
                  style: TextStyle(fontSize: 12, color: Color(0xFFA1A1AA)),
                ),
              ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: canApprove && !_busy ? _approve : null,
              icon: Icon(_busy ? Icons.hourglass_top : Icons.fingerprint),
              label: Text(_busy ? 'Waiting for authenticator…' : 'Approve with passkey'),
              style: FilledButton.styleFrom(
                backgroundColor: const Color(0xFF10B981),
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'A timeout denies the release. It never allows it.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 11, color: Color(0xFF52525B)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _row(String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: <Widget>[
            Text(label, style: const TextStyle(color: Color(0xFF52525B), fontSize: 12)),
            Flexible(
              child: Text(
                value,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.right,
                style: const TextStyle(color: Color(0xFFE4E4E7), fontSize: 13),
              ),
            ),
          ],
        ),
      );

  Widget _verdictPanel(ApprovalVerdict verdict) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: verdict.isAllowed
              ? const Color(0xFF064E3B)
              : const Color(0xFF881337),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              verdict.outcome == ApprovalOutcome.declassifiedAndAllowed
                  ? 'RELEASED'
                  : 'DENIED',
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            if (verdict.reason != null)
              Text(verdict.reason!.wire,
                  style: const TextStyle(fontSize: 12, color: Color(0xFFA1A1AA))),
          ],
        ),
      );
}