from typing import Any, Dict, List, Optional
import qnexus as qnx
from pytket.backends import Backend  # type: ignore
from .base import QuantumProvider

class QuantinuumProvider(QuantumProvider):
    """Quantinuum Nexus provider using qnexus + pytket integration."""

    def __init__(self):
        super().__init__()
        self._configured = False
        self._project = None
        self._client = None  # internal qnexus client if needed

    @property
    def provider_name(self) -> str:
        return "quantinuum"

    @property
    def sdk_type(self) -> str:
        # Nexus + Quantinuum devices are pytket-first.
        return "pytket"

    def authenticate(self, **kwargs: Any) -> List[str]:
        """
        Authenticate with Nexus.

        Parameters
        ----------
        username : str, optional
            Username for non-interactive login.
        password : str, optional
            Password for non-interactive login.
        browser : bool, default False
            If True, use `qnx.login()` which opens a browser.
        project_name : str, default "Default"
            Project to create/use for Nexus operations.
        overwrite : bool, default False
            Re-login even if already authenticated.

        Returns
        -------
        list of str
            Available backend identifiers.
        """
        overwrite = bool(kwargs.get("overwrite", False))
        username = kwargs.get("username")
        password = kwargs.get("password")
        use_browser = bool(kwargs.get("browser", False))
        project_name = kwargs.get("project_name", "Default")

        if self._configured and not overwrite:
            # Already configured; return devices already visible to qnexus
            return self.list_available_devices()

        # Perform login using qnexus helpers
        try:
            if username and password:
                # qnexus provides a login_no_interaction variant in some releases
                try:
                    qnx.login_no_interaction(user=username, pwd=password)  # docs show this exists
                except AttributeError:
                    # fallback to interactive version
                    qnx.login_with_credentials()
            elif use_browser:
                qnx.login()
            else:
                # Interactive fallback
                qnx.login_with_credentials()

            # set a project (Nexus requires a project scope for jobs)
            project = qnx.projects.get_or_create(name=project_name)
            qnx.context.set_active_project(project)
            self._project = project
            self._configured = True

            # Optionally cache a local client if needed
            try:
                from qnexus.client import get_nexus_client
                self._client = get_nexus_client()
            except Exception:
                self._client = None

            return self.list_available_devices()
        except Exception as e:
            raise RuntimeError(f"Quantinuum Nexus authentication failed: {e}") from e

    def list_available_devices(self, **kwargs) -> List[str]:
        """
        Query Nexus for available Quantinuum devices/backends.

        Returns
        -------
        list of str
            Device names or backend ids available in the account.
        """
        if not self._configured:
            raise RuntimeError("Call authenticate() before listing devices")

        # Nexus exposes backend configs and device listings via qnexus client apis.
        devices = []
        try:
            # The docs expose device-related methods under qnx.client.devices.*
            # We'll try to use a high-level query that returns available device names
            # (adjust the exact call if your qnexus version exposes a differently-named helper)
            from qnexus.client import devices as client_devices  # type: ignore
            # client_devices.get_all() isn't guaranteed to be the exact name; use defensive code:
            if hasattr(client_devices, "get_all"):
                devs = client_devices.get_all()
                devices = [d.name for d in devs]
            else:
                # Fallback: some docs mention BackendConfig listings; we'll fetch configs
                from qnexus.client import backend_configs as bc
                if hasattr(bc, "get_all"):
                    bcs = bc.get_all()
                    devices = [getattr(x, "name", str(x)) for x in bcs]
        except Exception:
            # Last-resort: return a small set of commonly-known Quantinuum targets if query failed
            devices = ["H2-Emulator", "H2-1", "H1-1"]
        return devices

    def add_device(self, device: str, alias: Optional[str] = None, **kwargs) -> Any:
        """
        Register a Nexus BackendConfig or pytket QuantinuumBackend in the provider registry.

        If qnexus is installed and `use_pytket_backend` is True in kwargs,
        create a pytket QuantinuumBackend; otherwise store a QuantinuumConfig reference.
        """
        if not self._configured:
            raise RuntimeError("Call authenticate() before adding devices")

        name = alias or device
        try:
            # Create a BackendConfig reference for this device using qnexus helper constructors
            # (docs show a `QuantinuumConfig` dataclass / factory)
            config_kwargs = kwargs.get("config_kwargs", {})
            quant_cfg = qnx.QuantinuumConfig(device_name=device, **config_kwargs)
            store_obj = {"config": quant_cfg}

            if kwargs.get("use_pytket_backend", True):
                # try to instantiate pytket QuantinuumBackend for local access if available
                try:
                    from pytket.extensions.quantinuum import QuantinuumBackend  # type: ignore
                    # The pytket QuantinuumBackend accepts a device string per docs
                    qt_backend = QuantinuumBackend(device_name=device)
                    store_obj["pytket_backend"] = qt_backend
                except Exception:
                    # If qnexus is not installed, we still keep the BackendConfig
                    pass

            self._backends[name] = store_obj
            return store_obj
        except Exception as e:
            raise RuntimeError(f"Failed to add Quantinuum device '{device}': {e}") from e

    def init_device(self, device: str, alias: Optional[str] = None, **kwargs: Any) -> str:
        """
        Authenticate if needed, set project, and add device. Returns alias.
        Accepts any kwargs accepted by authenticate() and add_device().
        """
        if not self._configured:
            # Try to use any provided credentials in kwargs
            self.authenticate(**kwargs)
        alias = alias or device
        self.add_device(device, alias, **kwargs)
        return alias

    # The base class has get_backend, remove_backend, list_backends, etc. which already operate on _backends.
    # For job-level operations you may want to add helper methods that call qnexus.jobs.* directly.
    # Example helper: submit a compile + execute flow (pseudocode):
    def compile_and_execute(
        self,
        backend_alias: str,
        circuits: List[Any],  # typically pytket.Circuit objects or circuit refs
        compile_kwargs: Optional[Dict[str, Any]] = None,
        execute_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        High-level helper that:
          1) ensures backend is registered
          2) uploads circuit(s) if needed
          3) submits a CompileJob
          4) waits for completion
          5) submits an ExecuteJob
          6) waits and returns BackendResult(s)
        """
        if backend_alias not in self._backends:
            raise ValueError(f"Backend alias '{backend_alias}' not found")
        store = self._backends[backend_alias]
        backend_config = store.get("config")

        compile_kwargs = compile_kwargs or {}
        execute_kwargs = execute_kwargs or {}

        # Upload circuits if they are pytket Circuits (qnexus requires circuits in its DB)
        refs = []
        for c in circuits:
            # if already a qnexus circuit reference, keep it
            if getattr(c, "__class__", None) and c.__class__.__name__ == "CircuitRef":
                refs.append(c)
            else:
                # upload pytket Circuit to Nexus DB
                ref = qnx.circuits.upload(circuit=c)
                refs.append(ref)

        # Submit compile job
        compile_job = qnx.jobs.compile(
            circuit_refs=refs,
            backend_config=backend_config,
            **compile_kwargs,
        )
        # Wait for compile completion
        qnx.jobs.wait_for(compile_job)
        compiled_outputs = qnx.jobs.results(compile_job)
        # take first compiled circuit as example
        compiled_ref = compiled_outputs[0].get_output()
        compiled_circuit = compiled_ref.download_circuit()

        # Submit execute job
        execute_job = qnx.jobs.execute(
            compiled_circuit_refs=[compiled_ref],
            backend_config=backend_config,
            **execute_kwargs,
        )
        qnx.jobs.wait_for(execute_job)
        results = qnx.jobs.results(execute_job)
        # results are BackendResult references; download / convert as needed
        return {"compile_job": compile_job, "execute_job": execute_job, "results": results}