# DeepStreaks (ds) + ZTF Solar System Marshal (ztfssm)

## Setting up DeepStreaks

See [readme.md](readme.md)::Sentinel service::Set-up instructions.

If DS gets stalled for no apparent reason (happens once in ~3-6 months),
restart the `deep-asteroids` container:

```shell
docker stop deep-asteroids && docker start deep-asteroids
```

## Transferring ztfssm from one machine to another

On the source machine, export the docker container with the marshal, e.g.:

```shell
docker export ztfsolarsystemmarshal_ztfssm_1 > ztfsolarsystemmarshal_ztfssm_1.tar
```

Scp the resulting tarball and the contents of the ZTFSolarSystemMarshal/SSMData 
and ZTFSolarSystemMarshal/SSMWeb folders onto the target machine. 
Beware of the potentially large size of the folders.

Import the snapshot into a tagged docker image:

```shell
docker import ztfsolarsystemmarshal_ztfssm_1.tar ztfsolarsystemmarshal_ztfssm:latest
```

Spin up the image properly mounting the necessary folders and mapping the web server port, e.g.:

```shell
docker run --name ztfsolarsystemmarshal_ztfssm -d -p 80:80 \
    -v /local/home/ztfss/streaks:/data \
    -v /local/home/ztfdd/dev/ztfssm/SSMWeb:/ssmweb \
    -v /local/home/ztfdd/dev/ztfssm/SSMData:/ssmdata \
    ztfsolarsystemmarshal_ztfssm:latest bash -c "service apache2 start"
```

If the apache server throws errors, exec into the running container and inspect the logs

```shell
docker exec -it ztfsolarsystemmarshal_ztfssm bash
vi /var/log/apache2/error.log
```

## Issues on the ztfss machine at IPAC

The system unnecessarily blocks most of the in/outbound traffic, 
which causes multiple problems. For example:

- Can't pip-install python packages into the system
- Some ZTFSSM's scripts don't function because external resources are unreachable,
  for example, www.tle.info. 

I cannot provide an explicit IP whitelist, because 
1) I don't know everything that is needed.
2) Underlying resource IPs may change.
